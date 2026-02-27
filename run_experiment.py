"""
IB-SparseAttention 完整实验脚本
================================

用法（三种场景）：

  # 场景 1：用内置合成数据快速验证（无 GPU 也可跑，约 2-5 分钟）
  python run_experiment.py --mode quick

  # 场景 2：用 distilgpt2 + LoRA 微调真实数据集（需要 GPU，约 30-60 分钟）
  python run_experiment.py --mode full --data adult

  # 场景 3：用大模型 + 4-bit 量化（节省显存，推荐 LLaMA-3 8B 等）
  python run_experiment.py --mode full --data adult --llm meta-llama/Meta-Llama-3-8B \
      --quantization 4bit --lora

环境要求：
  conda activate PAFT          # 或包含 be-great / torch / transformers 的环境
  pip install networkx          # Phase 2 图分析（已装则跳过）
  pip install bitsandbytes      # 仅量化模式需要
  pip install matplotlib        # 仅保存热力图时需要
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch

# ── 将 TIDE 目录加入路径，确保 ib_sparse_attention 可导入 ──────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ib_sparse_attention import IBSparseGReaT, compute_W_sparsity, log_W_statistics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_experiment")


# ─────────────────────────────────────────────────────────────────────────────
# 数据集加载
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(name: str) -> pd.DataFrame:
    """加载训练数据集。

    支持：
      'synthetic' — 内置 500 行合成数据（无网络依赖）
      'adult'     — UCI Adult Census（sklearn 自带，需联网首次下载）
      'diabetes'  — 皮马印第安人糖尿病数据集（sklearn 自带）
    """
    if name == "synthetic":
        return _make_synthetic_data()
    elif name == "adult":
        return _load_adult()
    elif name == "diabetes":
        return _load_diabetes()
    else:
        raise ValueError(f"Unknown dataset: {name!r}. Choose: synthetic / adult / diabetes")


def _make_synthetic_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """构造带有真实特征依赖关系的合成数据。

    依赖关系（便于验证 W 矩阵是否正确恢复）：
      income  ← education + age  （高学历年长者收入高）
      health  ← bmi + exercise   （低 BMI + 多运动者健康）
      city    ← income           （高收入者倾向大城市）
    """
    rng = np.random.default_rng(seed)
    n_rows = n

    age       = rng.integers(20, 65, n_rows).astype(float)
    education = rng.choice(["high_school", "bachelor", "master", "phd"], n_rows,
                            p=[0.4, 0.35, 0.15, 0.10])
    edu_score = {"high_school": 0, "bachelor": 1, "master": 2, "phd": 3}
    edu_num   = np.array([edu_score[e] for e in education])

    bmi      = rng.normal(26, 4, n_rows).clip(15, 45).round(1)
    exercise = rng.integers(0, 7, n_rows).astype(float)  # 每周运动天数

    # 有噪声的因果关系
    income_score = 20000 + age * 500 + edu_num * 8000 + rng.normal(0, 5000, n_rows)
    income = np.where(income_score < 30000, "low",
             np.where(income_score < 60000, "medium", "high"))

    health_score = 100 - (bmi - 22).clip(0) * 3 + exercise * 5 + rng.normal(0, 10, n_rows)
    health = np.where(health_score < 60, "poor",
             np.where(health_score < 80, "fair", "good"))

    city_prob = np.where(income == "high", 0.7,
                np.where(income == "medium", 0.4, 0.2))
    city = np.where(rng.random(n_rows) < city_prob, "urban", "rural")

    df = pd.DataFrame({
        "age": age,
        "education": education,
        "bmi": bmi,
        "exercise": exercise,
        "income": income,
        "health": health,
        "city": city,
    })
    logger.info(f"Synthetic dataset: {len(df)} rows, {len(df.columns)} features")
    return df


def _load_adult() -> pd.DataFrame:
    """加载 UCI Adult Census 数据集（分类任务常用基准）。"""
    from sklearn.datasets import fetch_openml
    logger.info("Downloading Adult Census dataset from OpenML...")
    data = fetch_openml("adult", version=2, as_frame=True)
    df = data.frame.copy()
    # 精简为 6 列，便于快速实验
    cols = ["age", "education-num", "hours-per-week",
            "occupation", "sex", "class"]
    df = df[cols].dropna().reset_index(drop=True)
    df.columns = ["age", "education_years", "hours_per_week",
                  "occupation", "sex", "income_class"]
    # 转换数值类型
    for c in ["age", "education_years", "hours_per_week"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    logger.info(f"Adult dataset: {len(df)} rows, {len(df.columns)} features")
    return df


def _load_diabetes() -> pd.DataFrame:
    """加载 Pima Indians Diabetes 数据集。"""
    from sklearn.datasets import load_diabetes as _ld
    data = _ld(as_frame=True)
    df = data.frame.copy()
    logger.info(f"Diabetes dataset: {len(df)} rows, {len(df.columns)} features")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 实验主函数
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    # ── 1. 加载数据 ──────────────────────────────────────────────────────────
    df_train = load_dataset(args.data)
    print(f"\n训练数据预览:\n{df_train.head(3)}\n")
    print(f"特征列表: {list(df_train.columns)}")
    print(f"数据规模: {df_train.shape}\n")

    # ── 2. 构建 IBSparseGReaT 模型 ────────────────────────────────────────────
    lora_cfg = None
    if args.lora:
        lora_cfg = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05}

    model = IBSparseGReaT(
        llm=args.llm,

        # IB 超参数
        beta_ib=args.beta_ib,
        lambda_sparse=args.lambda_sparse,
        w_lr=args.w_lr,

        # 三阶段 epoch 数
        phase1_epochs=args.phase1_epochs,
        phase3_epochs=args.phase3_epochs,

        # 量化（需要 bitsandbytes）
        quantization=args.quantization,   # None / "4bit" / "8bit"

        # LoRA（推荐大模型时开启）
        efficient_finetuning="lora" if args.lora else "",
        lora_config=lora_cfg,

        # 训练设置
        batch_size=args.batch_size,
        experiment_dir=args.output_dir,

        # 监控
        save_heatmaps=args.save_heatmaps,
        heatmap_dir=os.path.join(args.output_dir, "heatmaps"),

        # HuggingFace Trainer 透传参数
        logging_steps=5,
        save_strategy="epoch",
        report_to=[],             # 关闭 W&B
        fp16=args.fp16,
    )

    # ── 3. 三阶段训练 ─────────────────────────────────────────────────────────
    print("=" * 60)
    print("开始 IB-SparseAttention 三阶段训练")
    print("=" * 60)

    results = model.fit(
        data=df_train,
        conditional_col=df_train.columns[-1],   # 最后一列作为条件列
    )

    # ── 4. Phase 2 分析结果 ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Phase 2 结果：特征依赖图分析")
    print("=" * 60)

    optimal_order = results["optimal_order"]
    cols = list(df_train.columns)
    print(f"最优特征顺序（列索引）: {optimal_order}")
    print(f"最优特征顺序（列名称）: {[cols[i] for i in optimal_order]}")

    W_final = results["W"]  # cpu tensor, detached
    print(f"\nW 矩阵统计:")
    import torch.nn.functional as F
    W_pos = F.softplus(W_final.float())
    sparsity = compute_W_sparsity(W_final, threshold=0.5)
    print(f"  矩阵形状     : {W_pos.shape}")
    print(f"  稀疏度        : {sparsity:.3f}  (< 0.5 的比例)")
    print(f"  密度           : {1 - sparsity:.3f}")
    print(f"  最大值         : {W_pos.max().item():.4f}")
    print(f"  均值           : {W_pos.mean().item():.4f}")

    # 打印前 5 强依赖
    print("\n  最强特征依赖对（Top-5）:")
    m = W_pos.shape[0]
    flat = W_pos.numpy().flatten()
    top_idx = flat.argsort()[::-1]
    printed = 0
    for idx in top_idx:
        i, j = divmod(int(idx), m)
        if i == j:
            continue
        i_name = cols[i] if i < len(cols) else str(i)
        j_name = cols[j] if j < len(cols) else str(j)
        print(f"    W[{i_name:20s} → {j_name:20s}] = {flat[idx]:.4f}")
        printed += 1
        if printed >= 5:
            break

    # 保存 W 矩阵依赖表
    dep_df = model.get_dependency_matrix()
    dep_path = os.path.join(args.output_dir, "W_dependency_matrix.csv")
    dep_df.to_csv(dep_path)
    print(f"\nW 依赖矩阵已保存至: {dep_path}")

    # ── 5. 采样生成合成数据 ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("采样生成合成数据")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    syn_df = model.sample(
        n_samples=args.n_samples,
        temperature=0.7,
        max_length=200,
        device=device,
        guided_sampling=True,      # 使用 optimal_order 的引导采样
        random_feature_order=False,
    )

    print(f"\n生成数据预览（前 5 行）:")
    print(syn_df.head())
    print(f"\n生成数据形状: {syn_df.shape}")
    print(f"缺失值统计:\n{syn_df.isna().sum()}")

    # 保存生成数据
    syn_path = os.path.join(args.output_dir, "synthetic_data.csv")
    syn_df.to_csv(syn_path, index=False)
    print(f"\n合成数据已保存至: {syn_path}")

    # ── 6. 简单质量评估（如果有原始数值列）────────────────────────────────────
    num_cols = df_train.select_dtypes(include=[float, int]).columns.tolist()
    if num_cols and len(syn_df) > 0:
        print("\n" + "=" * 60)
        print("数值特征分布对比（真实 vs 生成）")
        print("=" * 60)
        for c in num_cols[:3]:  # 最多打印 3 列
            if c in syn_df.columns:
                real_vals = df_train[c].dropna()
                syn_vals = pd.to_numeric(syn_df[c], errors="coerce").dropna()
                if len(syn_vals) > 0:
                    print(f"\n  {c}:")
                    print(f"    真实: mean={real_vals.mean():.2f}, "
                          f"std={real_vals.std():.2f}, "
                          f"min={real_vals.min():.2f}, max={real_vals.max():.2f}")
                    print(f"    生成: mean={syn_vals.mean():.2f}, "
                          f"std={syn_vals.std():.2f}, "
                          f"min={syn_vals.min():.2f}, max={syn_vals.max():.2f}")

    print("\n实验完成！")
    print(f"输出目录: {os.path.abspath(args.output_dir)}")
    return model, syn_df


# ─────────────────────────────────────────────────────────────────────────────
# CLI 参数解析
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="IB-SparseAttention 实验脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── 运行模式 ──────────────────────────────────────────────────────────────
    p.add_argument(
        "--mode", choices=["quick", "full"], default="quick",
        help="quick: 合成数据+小 epoch 快速验证；full: 完整训练流程",
    )

    # ── 数据集 ────────────────────────────────────────────────────────────────
    p.add_argument(
        "--data", choices=["synthetic", "adult", "diabetes"], default="synthetic",
        help="训练数据集",
    )

    # ── 基础模型 ──────────────────────────────────────────────────────────────
    p.add_argument(
        "--llm", default="distilgpt2",
        help="HuggingFace 模型检查点（如 distilgpt2 / gpt2 / meta-llama/...）",
    )
    p.add_argument(
        "--quantization", choices=["4bit", "8bit"], default=None,
        help="bitsandbytes 量化精度（None=全精度）",
    )
    p.add_argument(
        "--lora", action="store_true",
        help="是否使用 LoRA 高效微调（推荐大模型时开启）",
    )
    p.add_argument("--fp16", action="store_true", help="启用 fp16 训练")

    # ── IB 超参数 ─────────────────────────────────────────────────────────────
    p.add_argument("--beta_ib",       type=float, default=0.05,
                   help="IB 损失权重 β")
    p.add_argument("--lambda_sparse", type=float, default=1e-3,
                   help="L1 稀疏正则化权重 λ")
    p.add_argument("--w_lr",          type=float, default=5e-3,
                   help="W 矩阵独立学习率")

    # ── 训练轮数（根据 mode 自动覆盖）────────────────────────────────────────
    p.add_argument("--phase1_epochs", type=int, default=None,
                   help="Phase 1 训练 epoch 数（None 则由 mode 决定）")
    p.add_argument("--phase3_epochs", type=int, default=None,
                   help="Phase 3 训练 epoch 数（None 则由 mode 决定）")
    p.add_argument("--batch_size",    type=int, default=4)

    # ── 采样 ──────────────────────────────────────────────────────────────────
    p.add_argument("--n_samples", type=int, default=100,
                   help="生成的合成样本数量")

    # ── 输出 ──────────────────────────────────────────────────────────────────
    p.add_argument("--output_dir",    default="ib_sparse_output",
                   help="检查点、W 矩阵、生成数据的保存目录")
    p.add_argument("--save_heatmaps", action="store_true",
                   help="每 epoch 保存 W 热力图 PNG（需要 matplotlib）")

    args = p.parse_args()

    # ── 根据 mode 设置默认 epoch ────────────────────────────────────────────
    if args.mode == "quick":
        args.phase1_epochs = args.phase1_epochs or 2   # 快速验证
        args.phase3_epochs = args.phase3_epochs or 1
        if args.data == "synthetic":
            pass  # 默认用合成数据
    else:  # full
        args.phase1_epochs = args.phase1_epochs or 20
        args.phase3_epochs = args.phase3_epochs or 10

    return args


if __name__ == "__main__":
    args = parse_args()

    print("\n" + "=" * 60)
    print("IB-SparseAttention 实验配置")
    print("=" * 60)
    for k, v in vars(args).items():
        print(f"  {k:20s}: {v}")
    print()

    os.makedirs(args.output_dir, exist_ok=True)
    run(args)

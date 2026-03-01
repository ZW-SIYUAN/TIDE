# -*- coding: utf-8 -*-
"""
US Location 鸟类数据集 IB-SparseAttention 实验
================================================
v0.2 (2026-03-01) — 加入 CategoricalAtomizer 插件，修复 token 前缀竞争问题

数据集：data/us_location_train.csv
  列名：state_code | lat | lon | bird | lat_zone
  规模：16,320 行，5 个特征

期望 W 矩阵应恢复的真实依赖结构（Ground Truth，来自 HYFD/HANE）：
  lat  ──→ lat_zone      （纬度决定纬度区）
  lat  ──→ state_code    │
  lon  ──→ state_code    ├─ 坐标决定州代码
  lat  ──→ bird          │
  lon  ──→ bird          │
  state_code ──→ bird    （州→鸟种）

v0.1 已知问题（见 CHANGELOG.md）：
  - 13 个州合成量为 0（训练集均有 ~320 条）
  - 根因 A：western lon + southern lat 联合分布未学到（CA/NV）
  - 根因 B："Northern Cardinal" 被 "Northern Mockingbird" token 竞争压制
  v0.2 修复根因 B；根因 A 通过 --temperature 1.0 缓解。

用法：
  # 快速验证（v0.2 新架构）
  python experiment_us_location.py --mode quick

  # 完整实验（v0.2）
  python experiment_us_location.py --mode full

  # 消融对照：关闭 atomizer，复现 v0.1 基线
  python experiment_us_location.py --mode full --no-atomizer

  # LoRA + 4bit 量化（显存受限时）
  python experiment_us_location.py --mode full --lora --quantization 4bit

参数变更说明（v0.1 → v0.2）：
  max_length  : 120 → 50   测量值：原始行最长 36 token，原子化后最长 31 token
                            原来 120 严重过大，浪费推理时间
  temperature : 0.7 → 1.0  0.7 使 lon/lat 分布收窄（std 缩减 20-31%），
                            是导致 CA/NV 缺失的工程原因之一；
                            1.0 恢复标准 softmax，匹配训练分布
"""

import argparse
import logging
import os
import sys
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ib_sparse_attention import IBSparseGReaT, compute_W_sparsity
from atomic_tokenizer import CategoricalAtomizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("us_location_exp")

# ─────────────────────────────────────────────────────────────────────────────
# 数据加载与预处理
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TRAIN_CSV = os.path.join(DATA_DIR, "us_location_train.csv")
TEST_CSV  = os.path.join(DATA_DIR, "us_location_test.csv")

# 已知的真实特征依赖对（用于验证 W 矩阵）
# (i, j) 表示特征 i 影响特征 j
GROUND_TRUTH_EDGES = {
    ("lat", "lat_zone"),
    ("lat", "state_code"),
    ("lon", "state_code"),
    ("lat", "bird"),
    ("lon", "bird"),
    ("state_code", "bird"),
}

COLUMNS = ["state_code", "lat", "lon", "bird", "lat_zone"]


def load_train_data(subsample: int = 0) -> pd.DataFrame:
    """加载训练集，可选采样以加速实验。

    Args:
        subsample: 若 > 0，随机采样该行数（quick 模式用 2000 行即可）。

    Returns:
        处理好的 DataFrame（lat/lon 保留 2 位小数）。
    """
    df = pd.read_csv(TRAIN_CSV)
    # 精简浮点精度，减少 token 数量（每行文本更短 → 更快训练）
    df["lat"] = df["lat"].round(2)
    df["lon"] = df["lon"].round(2)
    if subsample > 0 and subsample < len(df):
        df = df.sample(n=subsample, random_state=42).reset_index(drop=True)
    logger.info(f"训练集: {len(df)} 行, 列: {list(df.columns)}")
    return df


def load_test_data() -> pd.DataFrame:
    df = pd.read_csv(TEST_CSV)
    df["lat"] = df["lat"].round(2)
    df["lon"] = df["lon"].round(2)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 评估函数
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_column_coverage(
    syn_df: pd.DataFrame,
    real_df: pd.DataFrame,
    cat_cols: list,
    num_cols: list,
) -> dict:
    """评估生成数据的列级质量。

    分类列：计算覆盖率（生成值在真实值域内的比例）
    数值列：计算均值/标准差的相对误差
    """
    metrics = {}
    for col in cat_cols:
        if col not in syn_df.columns:
            continue
        real_vals = set(real_df[col].dropna().unique())
        syn_vals  = syn_df[col].dropna()
        coverage  = syn_vals.isin(real_vals).mean()
        metrics[f"{col}_coverage"] = float(coverage)
        logger.info(f"  [{col}] coverage = {coverage:.3f}")

    for col in num_cols:
        if col not in syn_df.columns:
            continue
        real_series = real_df[col].dropna()
        syn_series  = pd.to_numeric(syn_df[col], errors="coerce").dropna()
        if len(syn_series) == 0:
            continue
        rmean = real_series.mean();  smean = syn_series.mean()
        rstd  = real_series.std();   sstd  = syn_series.std()
        mean_err = abs(smean - rmean) / (abs(rmean) + 1e-8)
        std_err  = abs(sstd  - rstd ) / (abs(rstd)  + 1e-8)
        metrics[f"{col}_mean_err"] = float(mean_err)
        metrics[f"{col}_std_err"]  = float(std_err)
        logger.info(
            f"  [{col}] real mean={rmean:.2f}±{rstd:.2f}  "
            f"syn mean={smean:.2f}±{sstd:.2f}  "
            f"mean_err={mean_err:.3f}"
        )
    return metrics


def _detect_bimodal_threshold(vals: np.ndarray) -> float:
    """在双峰分布中自动找分割阈值（最大间隔法）。

    W 矩阵经 softplus 后呈明显双峰：
      - 被压制边：接近 0（~1e-11 经 softplus ≈ 0.69, 原始 W 很负时 ≈ 0）
      - 激活边  ：约 25
    通过排序后找最大跳变点定阈值，比固定分位数更鲁棒。
    """
    sorted_vals = np.sort(vals)
    gaps = np.diff(sorted_vals)
    split_idx = int(np.argmax(gaps))
    threshold = (sorted_vals[split_idx] + sorted_vals[split_idx + 1]) / 2.0
    return float(threshold)


def evaluate_W_against_ground_truth(
    W_pos: np.ndarray,
    col_names: list,
) -> dict:
    """将 W 矩阵与已知的真实依赖关系对比。

    判定标准：自动检测双峰分布中的天然分割点（最大间隔法），
    认为 W_pos[i, j] > 阈值 的边被"激活"。

    返回 precision / recall / F1（忽略对角线）。
    """
    m = W_pos.shape[0]

    # 构造预测的依赖集合（排除对角线）
    flat = [(W_pos[i, j], i, j) for i in range(m) for j in range(m) if i != j]
    vals = np.array([v for v, _, _ in flat])

    # 双峰自动阈值（最大间隔法）
    threshold = _detect_bimodal_threshold(vals)
    n_active = int((vals > threshold).sum())

    predicted = {
        (col_names[i], col_names[j])
        for v, i, j in flat
        if v > threshold
    }

    tp = len(predicted & GROUND_TRUTH_EDGES)
    fp = len(predicted - GROUND_TRUTH_EDGES)
    fn = len(GROUND_TRUTH_EDGES - predicted)

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    logger.info(f"\n  W 矩阵 vs 真实依赖 (双峰自动阈值={threshold:.4f}, 激活边={n_active}/{len(flat)}):")
    logger.info(f"    TP={tp}  FP={fp}  FN={fn}")
    logger.info(f"    Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}")

    # 打印漏检的真实依赖
    missed = GROUND_TRUTH_EDGES - predicted
    if missed:
        logger.info(f"    漏检依赖: {missed}")
    false_pos = predicted - GROUND_TRUTH_EDGES
    if false_pos:
        logger.info(f"    误检依赖: {false_pos}")

    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "threshold": threshold}


# ─────────────────────────────────────────────────────────────────────────────
# 主实验流程
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. 加载数据 ──────────────────────────────────────────────────────────
    df_train = load_train_data(subsample=args.subsample)
    df_test  = load_test_data()

    # n_samples=0 → 自动对齐训练集大小
    if args.n_samples == 0:
        args.n_samples = len(df_train)
        logger.info(f"n_samples 自动设置为训练集大小: {args.n_samples}")

    cat_cols = ["state_code", "bird", "lat_zone"]
    num_cols = ["lat", "lon"]

    # ── v0.2: 原子化插件 ──────────────────────────────────────────────────────
    # --no-atomizer 复现 v0.1 基线（vanilla GReaT）
    # 默认开启，修复 token 前缀竞争（bird 列 30 值, state_code 2 值 → 32 新 token）
    if args.no_atomizer:
        atomizer = None
        logger.info("[AtomicTokenizer] 已禁用（--no-atomizer），使用 v0.1 baseline")
    else:
        atomizer = CategoricalAtomizer(cat_cols=["bird", "lat_zone", "state_code"])
        logger.info("[AtomicTokenizer] 已启用（v0.2 默认）")

    print(f"\n训练集预览:")
    print(df_train.head(4).to_string(index=False))
    print(f"\n各列信息:")
    for col in df_train.columns:
        if col in cat_cols:
            print(f"  {col:12s} 分类  {df_train[col].nunique():3d} 类")
        else:
            print(f"  {col:12s} 数值  [{df_train[col].min():.2f}, {df_train[col].max():.2f}]")

    # ── 2. 构建模型 ───────────────────────────────────────────────────────────
    lora_cfg = {"r": 8, "lora_alpha": 16, "lora_dropout": 0.05} if args.lora else None

    model = IBSparseGReaT(
        llm=args.llm,

        # v0.2: 原子化插件（None = 关闭，复现 v0.1 baseline）
        atomizer=atomizer,

        # IB 超参数
        beta_ib=args.beta_ib,
        lambda_sparse=args.lambda_sparse,
        w_lr=args.w_lr,

        # Phase 2 基数约束（高熵→低熵方向强制）
        cardinality_ratio_threshold=args.cardinality_ratio_threshold,

        # 三阶段 epoch
        phase1_epochs=args.phase1_epochs,
        phase3_epochs=args.phase3_epochs,

        # 量化
        quantization=args.quantization,

        # LoRA
        efficient_finetuning="lora" if args.lora else "",
        lora_config=lora_cfg,

        # 训练设置
        batch_size=args.batch_size,
        float_precision=2,          # lat/lon 保留 2 位小数
        experiment_dir=args.output_dir,

        # 监控
        save_heatmaps=args.save_heatmaps,
        heatmap_dir=os.path.join(args.output_dir, "W_heatmaps"),

        # Trainer 透传
        logging_steps=10,
        save_strategy="epoch",
        bf16=args.bf16,    # bf16 优先（Blackwell/Hopper/Ampere）
        fp16=args.fp16,    # 与 bf16 互斥，旧 GPU 备选
        report_to=[],
    )

    # ── 3. 三阶段训练 ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("IB-SparseAttention 三阶段训练")
    print("=" * 60)

    results = model.fit(
        data=df_train,
        conditional_col="bird",    # 以鸟种作为条件列（最终预测目标）
    )

    # ── 4. W 矩阵分析 ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Phase 2 结果：特征依赖结构分析")
    print("=" * 60)

    W_tensor = results["W"]                          # cpu fp32 tensor
    W_pos    = F.softplus(W_tensor).numpy()          # 实际调制值
    col_names = list(df_train.columns)               # 列名
    m = len(col_names)

    sparsity = compute_W_sparsity(W_tensor, threshold=0.5)
    print(f"\nW 矩阵稀疏度 (softplus(W) < 0.5): {sparsity:.3f}")
    print(f"W 矩阵密度:                         {1-sparsity:.3f}")

    # 打印完整 W 矩阵（行=预测者，列=被预测者）
    print("\nW 依赖强度矩阵（softplus 变换后）:")
    dep_df = model.get_dependency_matrix()
    print(dep_df.round(3).to_string())

    # 保存 W 矩阵 CSV
    w_csv_path = os.path.join(args.output_dir, "W_dependency_matrix.csv")
    dep_df.to_csv(w_csv_path, encoding="utf-8")
    print(f"\nW 矩阵已保存: {w_csv_path}")

    # 最优特征顺序
    optimal_order = results["optimal_order"]
    optimal_names = [col_names[i] for i in optimal_order]
    print(f"\n最优特征生成顺序: {optimal_names}")
    print(f"（索引）         : {optimal_order}")

    # 与 Ground Truth 比对
    print("\n与已知真实依赖结构对比:")
    w_metrics = evaluate_W_against_ground_truth(W_pos, col_names)

    # 保存 W 相关指标
    w_info = {
        "optimal_order_names": optimal_names,
        "optimal_order_idx":   optimal_order,
        "W_sparsity":          float(sparsity),
        "W_density":           float(1 - sparsity),
        "w_precision":         w_metrics["precision"],
        "w_recall":            w_metrics["recall"],
        "w_f1":                w_metrics["f1"],
    }
    metrics_path = os.path.join(args.output_dir, "w_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(w_info, f, ensure_ascii=False, indent=2)
    print(f"\nW 矩阵指标已保存: {metrics_path}")

    # ── 5. 采样生成合成数据 ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("采样生成合成数据")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"推理设备: {device}")

    syn_df = model.sample(
        n_samples=args.n_samples,
        temperature=args.temperature,   # v0.2 默认 1.0（v0.1 旧值 0.7 会收窄 lat/lon 分布）
        max_length=50,                  # v0.2: 实测原始最长 36 token，原子化后 31 token；120 严重浪费
        device=device,
        guided_sampling=True,
        random_feature_order=False,     # 使用 Phase 2 推导的 optimal_order
    )

    print(f"\n生成数据（前 6 行）:")
    print(syn_df.head(6).to_string(index=False))
    print(f"\n生成数据统计:")
    print(f"  行数: {len(syn_df)}，列数: {len(syn_df.columns)}")
    print(f"  缺失值:\n{syn_df.isna().sum().to_string()}")

    # ── 6. 质量评估 ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("数据质量评估")
    print("=" * 60)

    print("\n--- 列级覆盖率 & 分布误差 ---")
    quality_metrics = evaluate_column_coverage(syn_df, df_train, cat_cols, num_cols)

    # 综合得分（覆盖率均值 + 数值误差均值）
    coverages = [v for k, v in quality_metrics.items() if k.endswith("_coverage")]
    mean_errs = [v for k, v in quality_metrics.items() if k.endswith("_mean_err")]
    if coverages:
        print(f"\n  平均分类覆盖率: {np.mean(coverages):.3f}")
    if mean_errs:
        print(f"  平均数值均值误差: {np.mean(mean_errs):.3f}")

    # ── 7. 保存结果 ───────────────────────────────────────────────────────────
    syn_path = os.path.join(args.output_dir, "synthetic_us_location.csv")
    syn_df.to_csv(syn_path, index=False, encoding="utf-8")
    print(f"\n合成数据已保存: {syn_path}")

    # 汇总 metrics
    all_metrics = {**quality_metrics, **w_info}
    summary_path = os.path.join(args.output_dir, "experiment_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)

    print(f"\n实验摘要已保存: {summary_path}")
    print("\n" + "=" * 60)
    print("实验完成！")
    print(f"输出目录: {os.path.abspath(args.output_dir)}")
    print("=" * 60)

    return model, syn_df, all_metrics


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="US Location 数据集 IB-SparseAttention 实验",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", choices=["quick", "full"], default="quick",
                   help="quick=小样本快速验证; full=完整训练")
    p.add_argument("--llm",  default="gpt2-medium",
                   help="HuggingFace 模型检查点（默认 gpt2-medium）")
    p.add_argument("--lora", action="store_true",
                   help="使用 LoRA 微调（推荐大模型）")

    prec = p.add_mutually_exclusive_group()
    prec.add_argument("--bf16", action="store_true",
                      help="bf16 混合精度（推荐 Ampere/Hopper/Blackwell，如 RTX 5080）")
    prec.add_argument("--fp16", action="store_true",
                      help="fp16 混合精度（适合旧 Turing 卡，需 GradScaler）")

    p.add_argument("--quantization", choices=["4bit", "8bit"], default=None,
                   help="bitsandbytes 量化（None=全精度）")

    # IB 超参
    p.add_argument("--beta_ib",       type=float, default=0.05)
    p.add_argument("--lambda_sparse", type=float, default=5e-4)
    p.add_argument("--w_lr",          type=float, default=5e-3)

    # Phase 2 基数约束阈值
    # nunique[i] / nunique[j] > 此值时，强制方向为 i→j，斩断反向边
    # 默认 10（一个数量级差异即强制方向）；设为 inf 可完全禁用
    p.add_argument("--cardinality_ratio_threshold", type=float, default=10.0,
                   help="nunique 比值阈值，高于此值时强制高熵→低熵方向（default: 10）")

    # 训练轮数
    p.add_argument("--phase1_epochs", type=int, default=None)
    p.add_argument("--phase3_epochs", type=int, default=None)
    p.add_argument("--batch_size",    type=int, default=16)

    # 数据采样（quick 模式默认 2000 行）
    p.add_argument("--subsample", type=int, default=0,
                   help="采样行数（0=使用全量数据）")

    # v0.2: 原子化开关（默认开启；--no-atomizer 复现 v0.1 基线）
    p.add_argument("--no-atomizer", action="store_true",
                   help="关闭 CategoricalAtomizer，复现 v0.1 vanilla GReaT 基线")

    # v0.2: 采样温度（0.7 收窄分布→CA/NV 缺失；1.0 匹配训练分布）
    p.add_argument("--temperature", type=float, default=1.0,
                   help="采样温度（default: 1.0；v0.1 旧值=0.7）")

    # 生成
    p.add_argument("--n_samples",    type=int, default=0,
                   help="生成的合成样本数量（0=自动对齐训练集大小）")

    # 输出
    p.add_argument("--output_dir",   default="output_us_location")
    p.add_argument("--save_heatmaps", action="store_true")

    args = p.parse_args()

    # 在 Blackwell / Hopper / Ampere GPU 上自动推荐 bf16
    if not args.fp16 and not args.bf16 and torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        if cap[0] >= 8:   # sm_80+（Ampere 及以上）支持原生 bf16 Tensor Core
            args.bf16 = True
            print(f"[INFO] 检测到 sm_{cap[0]}{cap[1]} GPU，自动启用 bf16（推荐）")

    # 根据 mode 补充默认值
    if args.mode == "quick":
        args.phase1_epochs = args.phase1_epochs or 3
        args.phase3_epochs = args.phase3_epochs or 2
        args.subsample     = args.subsample or 2000
        # quick 模式最多生成 50 行；n_samples=0 时也强制设为 50
        args.n_samples     = 50 if args.n_samples == 0 else min(args.n_samples, 50)
    else:  # full
        args.phase1_epochs = args.phase1_epochs or 30
        args.phase3_epochs = args.phase3_epochs or 10
        args.subsample     = args.subsample or 0   # 全量 16320 行
        # n_samples=0 留给 run_experiment 在加载数据后自动对齐训练集大小

    return args


if __name__ == "__main__":
    args = parse_args()

    print("=" * 60)
    print("实验配置")
    print("=" * 60)
    for k, v in vars(args).items():
        print(f"  {k:20s}: {v}")
    print()

    run_experiment(args)

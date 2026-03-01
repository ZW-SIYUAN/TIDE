# TIDE Framework — Changelog

每次架构更新必须包含：触发原因（实验结果）→ 根因分析 → 解决方案 → 修改文件。

---

## v0.2 — Atomic Categorical Tokenizer  `2026-03-01`

### 触发原因

US-Location 实验（`output_us_location/`）完整运行后，对比训练集与合成集的州分布：

- 训练集：51 个州，每州 ~320 条，变异系数 **CV = 2.4%**（近乎完美均衡）
- 合成集：仅 38 个州有数据，CV = **134%**（严重偏斜）
- **13 个州完全缺失**，合成量为 0，尽管训练集中每个州均有 ~320 条：
  `CA CT DC DE HI IN MD ME MI NV OH PA VA`

### 根因分析

经过逐步排查（见 `scatter_stats_synthetic.csv` 与坐标 z-score 分析），识别出**两种独立的失败模式**：

**模式 A — 地理分布偏差（CA、NV）**

生成顺序为 `lon → lat → state_code → bird → lat_zone`（IB Phase 2 自动推导）。
在 lon ≈ -120（西海岸经度）时，模型学到的条件分布 `P(lat | lon=-120)` 峰值在
lat ≈ 46–49（WA/OR 领域），导致 CA/NV 的纬度范围（lat 33–42）从不被采样。
WA 和 OR 的纬度刚好在均值以上，因此有大量样本；CA/NV 偏南，陷入分布尾部。

> 此模式本质是自回归生成的 **均值回归效应**（mean-seeking），叠加 lat std 从
> 6.42 收缩到 5.14（缩减 20%）后对尾部的额外压制。
> **工程层面**可通过提高采样温度（temperature: 0.7 → 1.0）缓解；
> 原理层面需要更多训练轮次或不同的特征顺序策略。

**模式 B — Token 前缀竞争（OH、IN、PA、VA、CT、MI 等 11 州）**

OH（lat z=+0.18, lon z=+0.90）的坐标几乎在合成分布的正中央，理论上最易生成，
但合成量为 **0**。根因在于鸟名：

| 鸟名 | 训练条数 | 合成条数 | 压制率 |
|------|---------|---------|--------|
| Northern **Cardinal** | 2551 | 7 | **99.7%** |
| Northern **Mockingbird** | — | 2524 | 主导 |
| American **Robin** | 969 | 1 | **99.9%** |

GReaT 将分类列的值编码为自然语言子串。当两个值共享 GPT-2 词表前缀时
（"Northern " → Cardinal / Mockingbird），频率更高的后缀几乎独占所有概率质量，
与其他列的条件上下文无关。这是**文本化表格生成的结构性缺陷**：
列值的表面文本相似度干扰了其统计独立性。

### 解决方案

新增独立模块 `atomic_tokenizer.py`，实现 `CategoricalAtomizer` 类。

**核心思路**：将每个需要多个 GPT-2 sub-token 的分类列取值注册为单个特殊 token。

```
训练前：bird 列值 "Northern Cardinal"   → 原始文本 → 3 sub-tokens，共享前缀
转换后：bird 列值 "ATM_bird_3"          → 1 special token，独立竞争
采样后：ATM_bird_3                      → 反映射回 "Northern Cardinal"
```

转换后，"ATM_bird_3" 与 "ATM_bird_8" 在 logit 空间中地位完全平等，
由且仅由上下文（其他列的值）决定哪个被选中，消除前缀干扰。

### 设计原则（积木化）

| 原则 | 实现 |
|------|------|
| 即插即用 | `atomizer=None` 默认关闭，传入实例即激活，**零侵入** ib_sparse_attention.py 内部逻辑 |
| 消融对照 | `atomizer=None` 复现 vanilla GReaT，`atomizer=CategoricalAtomizer()` 为新版本 |
| 选择性原子化 | 只处理 token 数 ≥ `min_tokens`（默认 2）的值，单 token 值无需变更 |
| 透明性 | 原子 token 是单个 token，与 `FeatureAlignmentDataset` 的 offset-based `feature_map` 完全兼容 |
| 持久化 | `save()` / `load()` 保存映射表，支持跨实验复用 |

### 修改文件

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `atomic_tokenizer.py` | **新建** | CategoricalAtomizer、diagnose()、save/load |
| `ib_sparse_attention.py` | 微改（3处） | `__init__` 加 `atomizer` 参数；`fit()` 调用 `fit_transform` + `resize_token_embeddings`；`sample()` 调用 `inverse_transform` |
| `CHANGELOG.md` | **新建** | 本文件 |

### 使用方式

```python
from atomic_tokenizer import CategoricalAtomizer

# 开启原子化（新架构）
atomizer = CategoricalAtomizer(cat_cols=["bird", "state_code", "lat_zone"])
model = IBSparseGReaT(llm="gpt2-medium", atomizer=atomizer, ...)

# 消融对照（vanilla GReaT 行为）
model_baseline = IBSparseGReaT(llm="gpt2-medium", atomizer=None, ...)
```

### 预期效果

- 模式 B 失败的 11 个州（OH/IN/PA/VA 等）应重新出现在合成数据中
- Northern Cardinal / American Robin 的生成比例应接近训练集占比
- 模式 A（CA/NV）需配合温度调整或更多训练轮次，原子化对此无直接改善

---

## v0.1 — 初始实现  `2026-02-27`

**文件**：`ib_sparse_attention.py`

初始架构，实现了完整的 IB-SparseAttention 三阶段训练框架：

- `FeatureAlignmentDataset` / `FeatureAlignmentCollator` — 基于 offset_mapping 的特征对齐
- `IBSparseAttentionModulator` — monkey-patch F.scaled_dot_product_attention 注入 W 偏置
- `IBLossComputer` — 批量盲上下文 MI 估计
- `IBSparseTrainer` — L_CE + β·MI + λ·‖W‖₁ 联合损失
- `IBSparseGReaT` — 三阶段 fit()：Phase1（随机顺序，训练 W+LoRA）→ Phase2（networkx 最大生成树推导顺序）→ Phase3（固定顺序，仅训练 LoRA）

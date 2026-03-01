# -*- coding: utf-8 -*-
"""
从已有的 W_dependency_matrix.csv 生成热力图并保存为 PNG。

用法：
  python plot_w_heatmap.py
  python plot_w_heatmap.py --csv output_us_location/W_dependency_matrix.csv
  python plot_w_heatmap.py --csv output_us_location/W_dependency_matrix.csv --out w_heatmap.png
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # 无头模式，不弹窗
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Ground truth edges（已去除 lat_zone->bird）
GROUND_TRUTH_EDGES = {
    ("lat", "lat_zone"),
    ("lat", "state_code"),
    ("lon", "state_code"),
    ("lat", "bird"),
    ("lon", "bird"),
    ("state_code", "bird"),
}


def detect_bimodal_threshold(vals: np.ndarray) -> float:
    """最大间隔法自动找双峰分割点。"""
    sorted_vals = np.sort(vals)
    gaps = np.diff(sorted_vals)
    split_idx = int(np.argmax(gaps))
    return float((sorted_vals[split_idx] + sorted_vals[split_idx + 1]) / 2.0)


def plot_heatmap(csv_path: str, out_path: str):
    # 读取 CSV
    dep_df = pd.read_csv(csv_path, index_col=0)
    col_names = list(dep_df.columns)
    m = len(col_names)
    W_pos = dep_df.values.astype(float)   # shape (m, m)

    # 对角线置 NaN（自身依赖无意义）
    W_display = W_pos.copy()
    np.fill_diagonal(W_display, np.nan)

    # 计算自动阈值
    off_diag = W_pos[~np.eye(m, dtype=bool)]
    threshold = detect_bimodal_threshold(off_diag)
    active_mask = W_display > threshold

    # ── 绘图 ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # ─── 左图：原始值热力图 ────────────────────────────────────────────────────
    ax = axes[0]
    # 压缩数值范围以显示双峰（对数缩放更清晰）
    W_log = np.log1p(W_display)   # log(1 + x)，NaN 保持 NaN
    vmax = np.nanmax(W_log)

    im = ax.imshow(W_log, cmap="RdYlGn", vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(m)); ax.set_xticklabels(col_names, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(m)); ax.set_yticklabels(col_names, fontsize=9)
    ax.set_title("W Dependency Strength (log scale)\nrow=source feature, col=target feature", fontsize=10)
    ax.set_xlabel("Target feature j")
    ax.set_ylabel("Source feature i")

    # 标注数值
    for i in range(m):
        for j in range(m):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", fontsize=8, color="gray")
            else:
                val = W_pos[i, j]
                txt = f"{val:.1f}" if val > 1 else f"{val:.1e}"
                color = "white" if W_log[i, j] > vmax * 0.6 else "black"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=color)

    plt.colorbar(im, ax=ax, label="log(1 + softplus(W))", fraction=0.046, pad=0.04)


    # ─── 右图：二值化热力图（激活 vs 压制）+ GT 标注 ─────────────────────────
    ax2 = axes[1]
    binary = np.zeros((m, m))
    binary[active_mask] = 1.0
    binary[np.eye(m, dtype=bool)] = np.nan

    ax2.imshow(binary, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto", alpha=0.85)
    ax2.set_xticks(range(m)); ax2.set_xticklabels(col_names, rotation=40, ha="right", fontsize=9)
    ax2.set_yticks(range(m)); ax2.set_yticklabels(col_names, fontsize=9)
    ax2.set_title(f"W Binary Map (threshold={threshold:.2f})\nGT=red dashed  TP=green  FP=orange", fontsize=10)
    ax2.set_xlabel("Target feature j")
    ax2.set_ylabel("Source feature i")

    name2idx = {n: i for i, n in enumerate(col_names)}

    tp_count = fp_count = fn_count = 0
    for i in range(m):
        for j in range(m):
            if i == j:
                ax2.text(j, i, "—", ha="center", va="center", fontsize=8, color="gray")
                continue
            is_pred = bool(active_mask[i, j])
            is_gt   = (col_names[i], col_names[j]) in GROUND_TRUTH_EDGES
            label = "1" if is_pred else "0"
            ax2.text(j, i, label, ha="center", va="center", fontsize=9,
                     color="white" if is_pred else "black")

            # 标注框
            if is_gt and is_pred:   # TP
                rect = mpatches.FancyBboxPatch(
                    (j - 0.45, i - 0.45), 0.9, 0.9,
                    boxstyle="round,pad=0.05", linewidth=2.0,
                    edgecolor="lime", facecolor="none")
                ax2.add_patch(rect)
                tp_count += 1
            elif is_gt and not is_pred:  # FN
                rect = mpatches.FancyBboxPatch(
                    (j - 0.45, i - 0.45), 0.9, 0.9,
                    boxstyle="round,pad=0.05", linewidth=2.0,
                    edgecolor="red", facecolor="none", linestyle="--")
                ax2.add_patch(rect)
                fn_count += 1
            elif not is_gt and is_pred:  # FP
                rect = mpatches.FancyBboxPatch(
                    (j - 0.45, i - 0.45), 0.9, 0.9,
                    boxstyle="round,pad=0.05", linewidth=2.0,
                    edgecolor="orange", facecolor="none")
                ax2.add_patch(rect)
                fp_count += 1

    precision = tp_count / (tp_count + fp_count + 1e-8)
    recall    = tp_count / (tp_count + fn_count + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    legend_items = [
        mpatches.Patch(facecolor="green",  label=f"Active (softplus(W) > {threshold:.1f})"),
        mpatches.Patch(facecolor="white", edgecolor="gray", label="Suppressed"),
        mpatches.Patch(facecolor="none",  edgecolor="lime",   linewidth=2, label=f"TP={tp_count}"),
        mpatches.Patch(facecolor="none",  edgecolor="red",    linewidth=2, linestyle="--", label=f"FN={fn_count}"),
        mpatches.Patch(facecolor="none",  edgecolor="orange", linewidth=2, label=f"FP={fp_count}"),
    ]
    ax2.legend(handles=legend_items, loc="lower right", fontsize=7.5,
               framealpha=0.9, ncol=1)

    fig.suptitle(
        f"IB-SparseAttention W Matrix  |  P={precision:.2f}  R={recall:.2f}  F1={f1:.2f}",
        fontsize=12, fontweight="bold", y=1.01
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Heatmap saved: {os.path.abspath(out_path)}")
    print(f"  threshold={threshold:.4f}  TP={tp_count}  FP={fp_count}  FN={fn_count}")
    print(f"  Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="output_us_location/W_dependency_matrix.csv")
    p.add_argument("--out", default="output_us_location/W_heatmap.png")
    args = p.parse_args()
    plot_heatmap(args.csv, args.out)

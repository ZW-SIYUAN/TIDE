# -*- coding: utf-8 -*-
"""
plot_scatter_synthetic.py
=========================
Visualise IB-SparseAttention synthetic data (output_us_location/synthetic_us_location.csv)
against true US state boundaries.

Two figure modes
----------------
  panel   -- per-state scatter grid (default: top-N states by sample count)
  overview -- full-US choropleth of validity rate  +  scatter of all points

Usage
-----
  # Panel: top-4 states (default)
  python plot_scatter_synthetic.py

  # Panel: specific states
  python plot_scatter_synthetic.py --mode panel --states OK SD MA KS

  # Panel: all states in synthetic CSV
  python plot_scatter_synthetic.py --mode panel --states ALL --n-cols 5

  # Overview US map
  python plot_scatter_synthetic.py --mode overview

  # Both figures
  python plot_scatter_synthetic.py --mode both

  # Color by lat_zone or bird instead of validity
  python plot_scatter_synthetic.py --color lat_zone
  python plot_scatter_synthetic.py --color bird

Outputs (output_us_location/)
------------------------------
  scatter_panel_synthetic.png       -- per-state scatter grid
  scatter_overview_synthetic.png    -- full US choropleth + scatter
  scatter_stats_synthetic.csv       -- per-state validity statistics
"""

import argparse
import json
import logging
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.colorbar import ColorbarBase

try:
    from shapely.geometry import Point
    from shapely.geometry import shape as shapely_shape
    from shapely.ops import unary_union
except ImportError:
    print("ERROR: shapely is required.  pip install shapely")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
HERE        = Path(__file__).parent
OUTPUT_DIR  = HERE / "output_us_location"
OUTPUT_DIR.mkdir(exist_ok=True)

SYNTH_CSV   = OUTPUT_DIR / "synthetic_us_location.csv"

_GEOJSON_URL   = (
    "https://raw.githubusercontent.com/python-visualization/"
    "folium/master/examples/data/us-states.json"
)
_GEOJSON_CACHE = HERE / "us_states_boundaries.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("scatter_synthetic")

# ─────────────────────────────────────────────────────────────────────────────
# State code / name mapping
# ─────────────────────────────────────────────────────────────────────────────
ABBREV_TO_NAME: Dict[str, str] = {
    "AL": "Alabama",        "AK": "Alaska",         "AZ": "Arizona",
    "AR": "Arkansas",       "CA": "California",     "CO": "Colorado",
    "CT": "Connecticut",    "DE": "Delaware",        "FL": "Florida",
    "GA": "Georgia",        "HI": "Hawaii",          "ID": "Idaho",
    "IL": "Illinois",       "IN": "Indiana",         "IA": "Iowa",
    "KS": "Kansas",         "KY": "Kentucky",        "LA": "Louisiana",
    "ME": "Maine",          "MD": "Maryland",        "MA": "Massachusetts",
    "MI": "Michigan",       "MN": "Minnesota",       "MS": "Mississippi",
    "MO": "Missouri",       "MT": "Montana",         "NE": "Nebraska",
    "NV": "Nevada",         "NH": "New Hampshire",   "NJ": "New Jersey",
    "NM": "New Mexico",     "NY": "New York",        "NC": "North Carolina",
    "ND": "North Dakota",   "OH": "Ohio",            "OK": "Oklahoma",
    "OR": "Oregon",         "PA": "Pennsylvania",    "RI": "Rhode Island",
    "SC": "South Carolina", "SD": "South Dakota",    "TN": "Tennessee",
    "TX": "Texas",          "UT": "Utah",            "VT": "Vermont",
    "VA": "Virginia",       "WA": "Washington",      "WV": "West Virginia",
    "WI": "Wisconsin",      "WY": "Wyoming",
}
NAME_TO_ABBREV: Dict[str, str] = {v: k for k, v in ABBREV_TO_NAME.items()}

# lat_zone color palette
LAT_ZONE_PALETTE = {"low": "#e07b39", "middle": "#4c9be8", "high": "#6dbf67"}

# Validity colors
COLOR_VALID   = "#2196F3"   # blue
COLOR_INVALID = "#E53935"   # red


# ══════════════════════════════════════════════════════════════════════════════
# GeoJSON boundary loading
# ══════════════════════════════════════════════════════════════════════════════

def _download_geojson() -> None:
    log.info("Downloading US state boundaries ...")
    try:
        urllib.request.urlretrieve(_GEOJSON_URL, _GEOJSON_CACHE)
        log.info(f"  Cached -> {_GEOJSON_CACHE}")
    except Exception as exc:
        raise RuntimeError(
            f"Cannot download state boundaries: {exc}\n"
            f"Manually place us-states.json at {_GEOJSON_CACHE}"
        ) from exc


def load_all_geometries() -> Dict[str, object]:
    """Return {abbrev: shapely_geometry} for all states in GeoJSON."""
    if not _GEOJSON_CACHE.exists():
        _download_geojson()

    with open(_GEOJSON_CACHE, encoding="utf-8") as fh:
        geojson = json.load(fh)

    geometries: Dict[str, object] = {}
    for feat in geojson["features"]:
        name   = feat["properties"].get("name", "")
        abbrev = NAME_TO_ABBREV.get(name)
        if abbrev:
            geometries[abbrev] = shapely_shape(feat["geometry"])
    log.info(f"Loaded {len(geometries)} state geometries from GeoJSON")
    return geometries


# ══════════════════════════════════════════════════════════════════════════════
# Data loading & validity
# ══════════════════════════════════════════════════════════════════════════════

def load_synthetic(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["state_code"] = df["state_code"].astype(str).str.strip().str.upper()
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])
    # Sanity range filter
    df = df[(df["lat"].between(-90, 90)) & (df["lon"].between(-180, 180))]
    log.info(f"Loaded synthetic CSV: {len(df)} rows, {df['state_code'].nunique()} states")
    return df


def add_validity_column(df: pd.DataFrame, geometries: Dict[str, object]) -> pd.DataFrame:
    """Vectorised point-in-polygon per state."""
    valid_flags = np.zeros(len(df), dtype=bool)
    for abbrev, geom in geometries.items():
        mask = df["state_code"] == abbrev
        if not mask.any():
            continue
        sub   = df[mask]
        flags = np.array([
            geom.contains(Point(row.lon, row.lat)) or geom.touches(Point(row.lon, row.lat))
            for row in sub.itertuples()
        ])
        valid_flags[mask.values] = flags
    df = df.copy()
    df["valid"] = valid_flags
    return df


def compute_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for st, g in df.groupby("state_code"):
        nt = len(g)
        nv = int(g["valid"].sum())
        ni = nt - nv
        rows.append({
            "state_code":    st,
            "state_name":    ABBREV_TO_NAME.get(st, st),
            "n_total":       nt,
            "n_valid":       nv,
            "n_invalid":     ni,
            "pct_valid":     round(100.0 * nv / nt, 2),
            "pct_invalid":   round(100.0 * ni / nt, 2),
        })
    return pd.DataFrame(rows).sort_values("n_total", ascending=False).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# Drawing helpers
# ══════════════════════════════════════════════════════════════════════════════

def _draw_boundary(ax, geom, fc="#dceefb", ec="#1565c0", lw=1.8, alpha=0.35, zorder=1):
    polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    for poly in polys:
        x, y = poly.exterior.xy
        ax.fill(list(x), list(y), fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=zorder)
        ax.plot(list(x), list(y), color=ec, lw=lw, zorder=zorder + 1)
        for interior in poly.interiors:
            xi, yi = interior.xy
            ax.fill(list(xi), list(yi), fc="white", ec=ec, lw=0.8, alpha=1.0, zorder=zorder + 2)


def _auto_axis_limits(geom, lats: np.ndarray, lons: np.ndarray):
    bds    = geom.bounds          # (minx, miny, maxx, maxy)
    span   = max(bds[2] - bds[0], bds[3] - bds[1])
    margin = max(span * 0.45, 0.8)
    cap    = span * 2.5

    xlo = max(min(bds[0] - margin, lons.min() - 0.3), bds[0] - cap)
    xhi = min(max(bds[2] + margin, lons.max() + 0.3), bds[2] + cap)
    ylo = max(min(bds[1] - margin, lats.min() - 0.3), bds[1] - cap)
    yhi = min(max(bds[3] + margin, lats.max() + 0.3), bds[3] + cap)
    return xlo, xhi, ylo, yhi


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 – Panel (per-state scatter grid)
# ══════════════════════════════════════════════════════════════════════════════

def _scatter_one_state(
    ax, abbrev: str, sub: pd.DataFrame, geom,
    color_by: str = "validity",
) -> None:
    """Draw a single state panel."""
    lats = sub["lat"].values
    lons = sub["lon"].values
    valid = sub["valid"].values

    state_name = ABBREV_TO_NAME.get(abbrev, abbrev)
    nt = len(sub)
    ni = int((~valid).sum())
    nv = nt - ni
    pct_inv = 100.0 * ni / nt if nt > 0 else 0.0

    _draw_boundary(ax, geom)

    if color_by == "validity":
        if ni > 0:
            ax.scatter(lons[~valid], lats[~valid], c=COLOR_INVALID,
                       marker="x", s=60, linewidths=1.8, alpha=0.80, label=f"Invalid ({ni})", zorder=5)
        if nv > 0:
            ax.scatter(lons[valid],  lats[valid],  c=COLOR_VALID,
                       marker="o", s=20, alpha=0.55, label=f"Valid ({nv})", zorder=4)

    elif color_by == "lat_zone" and "lat_zone" in sub.columns:
        for zone, grp in sub.groupby("lat_zone"):
            c = LAT_ZONE_PALETTE.get(str(zone).lower(), "#888888")
            ax.scatter(grp["lon"], grp["lat"], c=c, marker="o",
                       s=20, alpha=0.60, label=zone, zorder=4)

    elif color_by == "bird" and "bird" in sub.columns:
        birds   = sub["bird"].unique()
        cmap    = cm.get_cmap("tab20", len(birds))
        b2c     = {b: cmap(i) for i, b in enumerate(birds)}
        for bird, grp in sub.groupby("bird"):
            ax.scatter(grp["lon"], grp["lat"], c=[b2c[bird]],
                       marker="o", s=20, alpha=0.60, label=bird, zorder=4)

    # Title color by severity
    tc = "#b71c1c" if pct_inv >= 60 else "#e65100" if pct_inv >= 30 else "#1b5e20"
    ax.set_title(
        f"{state_name} ({abbrev})\nInvalid: {ni}/{nt}  =  {pct_inv:.1f}%",
        fontsize=10, fontweight="bold", color=tc, pad=6,
    )
    ax.set_xlabel("Longitude", fontsize=8)
    ax.set_ylabel("Latitude",  fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.18, linewidth=0.4)

    # Legend: cap at 8 entries for bird/zone
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 8:
        handles, labels = handles[:8], labels[:8]
    if handles:
        ax.legend(handles, labels, fontsize=7, loc="best",
                  framealpha=0.85, markerscale=1.2, handlelength=1.5)

    xlo, xhi, ylo, yhi = _auto_axis_limits(geom, lats, lons)
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)


def figure_panel(
    df:         pd.DataFrame,
    geometries: Dict[str, object],
    target_states: List[str],
    save_path:  Path,
    n_cols:     int = 2,
    color_by:   str = "validity",
) -> None:
    matplotlib.rcParams.update({
        "font.family":    "DejaVu Sans",
        "font.size":      10,
        "axes.titlesize": 11,
        "figure.dpi":     150,
    })

    n = len(target_states)
    ncols = min(n, n_cols)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.5 * ncols, 7 * nrows))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    color_desc = {
        "validity": "blue = valid  |  red-x = invalid",
        "lat_zone": "orange = low  |  blue = middle  |  green = high",
        "bird":     "colored by bird species (top 8)",
    }.get(color_by, color_by)

    fig.suptitle(
        "IB-SparseAttention Synthetic Data  --  Coordinate Validity vs True State Boundaries\n"
        f"FD check: state_code \u2192 (lat, lon)     color: {color_desc}",
        fontsize=12, fontweight="bold", y=1.01,
    )

    for idx, abbrev in enumerate(target_states):
        ax   = axes_flat[idx]
        sub  = df[df["state_code"] == abbrev]
        geom = geometries[abbrev]
        _scatter_one_state(ax, abbrev, sub, geom, color_by=color_by)

    for i in range(n, len(axes_flat)):
        axes_flat[i].set_visible(False)

    # Global legend
    if color_by == "validity":
        legend_handles = [
            Line2D([], [], color=COLOR_INVALID, marker="x", markersize=9,
                   linestyle="None", markeredgewidth=1.8, label="Invalid (outside boundary)"),
            Line2D([], [], color=COLOR_VALID,   marker="o", markersize=8,
                   linestyle="None", label="Valid (inside boundary)"),
            mpatches.Patch(facecolor="#dceefb", edgecolor="#1565c0",
                           alpha=0.9, label="True state boundary"),
        ]
        fig.legend(handles=legend_handles, loc="lower center",
                   bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=9, frameon=True)

    # Footer annotation
    n_total   = len(df[df["state_code"].isin(target_states)])
    n_invalid = int((~df.loc[df["state_code"].isin(target_states), "valid"]).sum())
    pct_inv   = 100.0 * n_invalid / n_total if n_total > 0 else 0.0
    footer = (
        f"Synthetic CSV: {SYNTH_CSV.name}   |   "
        f"Shown: {n} states, {n_total} samples   |   "
        f"Overall invalid: {n_invalid} ({pct_inv:.1f}%)"
    )
    fig.text(0.5, -0.06, footer, ha="center", va="top", fontsize=8.5, color="#444444",
             bbox=dict(boxstyle="round,pad=0.4", fc="#f5f5f5", ec="#cccccc", alpha=0.9))

    plt.tight_layout(rect=[0, 0.0, 1, 1.0])
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Panel figure saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 – Overview (full US choropleth + scatter)
# ══════════════════════════════════════════════════════════════════════════════

def figure_overview(
    df:         pd.DataFrame,
    geometries: Dict[str, object],
    stats_df:   pd.DataFrame,
    save_path:  Path,
    color_by:   str = "validity",
) -> None:
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size":   10,
        "figure.dpi":  150,
    })

    # ── Layout: 1 large map + 1 colorbar strip + 1 bar chart ─────────────────
    fig = plt.figure(figsize=(20, 12))
    gs  = fig.add_gridspec(
        2, 2,
        width_ratios=[3.5, 1],
        height_ratios=[1, 0.06],
        hspace=0.12, wspace=0.05,
    )
    ax_map  = fig.add_subplot(gs[0, 0])
    ax_bar  = fig.add_subplot(gs[0, 1])
    ax_cbar = fig.add_subplot(gs[1, 0])

    # ── Choropleth: validity rate per state ───────────────────────────────────
    cmap_choro = cm.RdYlGn   # red=invalid, green=valid
    norm_choro = mcolors.Normalize(vmin=0, vmax=100)
    stat_dict  = dict(zip(stats_df["state_code"], stats_df["pct_valid"]))

    # Draw ALL US states (greyed if not in synthetic data)
    for abbrev, geom in geometries.items():
        if abbrev in stat_dict:
            pct  = stat_dict[abbrev]
            fc   = cmap_choro(norm_choro(pct))
            ec   = "#555555"
            lw   = 0.8
        else:
            fc   = "#eeeeee"
            ec   = "#aaaaaa"
            lw   = 0.5
        polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
        for poly in polys:
            x, y = poly.exterior.xy
            ax_map.fill(list(x), list(y), fc=fc, ec=ec, lw=lw, alpha=0.85, zorder=1)
            ax_map.plot(list(x), list(y), color=ec, lw=lw, zorder=2)

    # ── Scatter all synthetic points ──────────────────────────────────────────
    # Exclude Alaska from default scatter view to keep map clean
    df_cont = df[~df["state_code"].isin(["AK", "HI"])]
    df_ak   = df[df["state_code"] == "AK"]

    if color_by == "validity":
        inv_mask = ~df_cont["valid"].values
        val_mask =  df_cont["valid"].values
        if inv_mask.any():
            ax_map.scatter(
                df_cont.loc[inv_mask, "lon"], df_cont.loc[inv_mask, "lat"],
                c=COLOR_INVALID, marker="x", s=10, linewidths=0.9,
                alpha=0.55, zorder=5, label="Invalid",
            )
        if val_mask.any():
            ax_map.scatter(
                df_cont.loc[val_mask, "lon"], df_cont.loc[val_mask, "lat"],
                c=COLOR_VALID, marker="o", s=5, alpha=0.30, zorder=4, label="Valid",
            )
    elif color_by == "lat_zone" and "lat_zone" in df_cont.columns:
        for zone, grp in df_cont.groupby("lat_zone"):
            c = LAT_ZONE_PALETTE.get(str(zone).lower(), "#888888")
            ax_map.scatter(grp["lon"], grp["lat"], c=c, marker="o",
                           s=5, alpha=0.35, zorder=4, label=zone)

    # State abbreviation labels for states with data
    for abbrev, geom in geometries.items():
        if abbrev not in stat_dict or abbrev in ("AK", "HI"):
            continue
        cx, cy = geom.centroid.x, geom.centroid.y
        pct = stat_dict[abbrev]
        ax_map.text(
            cx, cy, f"{abbrev}\n{pct:.0f}%",
            ha="center", va="center", fontsize=5.5,
            fontweight="bold", color="#111111", zorder=8,
        )

    # Contiguous US extent (exclude AK/HI)
    ax_map.set_xlim(-130, -65)
    ax_map.set_ylim(23, 52)
    ax_map.set_xlabel("Longitude", fontsize=9)
    ax_map.set_ylabel("Latitude",  fontsize=9)
    ax_map.set_title(
        "Synthetic Data  --  Coordinate Validity Choropleth + Scatter\n"
        "Each state colored by % valid; state abbreviation + validity% labeled",
        fontsize=11, fontweight="bold",
    )
    ax_map.tick_params(labelsize=8)
    ax_map.grid(True, alpha=0.15, linewidth=0.4)

    # Validity scatter legend
    if color_by == "validity":
        ax_map.legend(
            handles=[
                Line2D([], [], color=COLOR_INVALID, marker="x", markersize=8,
                       linestyle="None", markeredgewidth=1.5, label="Invalid point"),
                Line2D([], [], color=COLOR_VALID,   marker="o", markersize=7,
                       linestyle="None", label="Valid point"),
            ],
            loc="lower left", fontsize=8, framealpha=0.85,
        )

    # ── Colorbar: % valid ─────────────────────────────────────────────────────
    cb = ColorbarBase(ax_cbar, cmap=cmap_choro, norm=norm_choro,
                      orientation="horizontal")
    cb.set_label("% Valid Coordinates  (green=100% valid, red=0% valid)", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    # ── Bar chart: top-15 states by n_total, colored by % valid ──────────────
    top15 = stats_df.head(15).copy()
    colors_bar = [cmap_choro(norm_choro(p)) for p in top15["pct_valid"]]
    bars = ax_bar.barh(
        top15["state_code"][::-1], top15["n_total"][::-1],
        color=colors_bar[::-1], edgecolor="#444444", linewidth=0.6, height=0.7,
    )
    # Annotate validity %
    for bar, (_, row) in zip(bars, top15[::-1].iterrows()):
        x = bar.get_width()
        ax_bar.text(
            x + max(top15["n_total"]) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{row['pct_valid']:.0f}%",
            va="center", ha="left", fontsize=7.5, color="#333333",
        )
    ax_bar.set_xlabel("Sample Count", fontsize=9)
    ax_bar.set_title("Top 15 States\n(bar = count, color = %valid)", fontsize=9, fontweight="bold")
    ax_bar.tick_params(axis="y", labelsize=8)
    ax_bar.tick_params(axis="x", labelsize=7)
    ax_bar.grid(axis="x", alpha=0.25, linewidth=0.5)
    ax_bar.spines[["top", "right"]].set_visible(False)

    # ── Super title ───────────────────────────────────────────────────────────
    n_total   = len(df)
    n_invalid = int((~df["valid"]).sum())
    n_states  = df["state_code"].nunique()
    fig.suptitle(
        f"IB-SparseAttention Synthetic US-Location Data  |  "
        f"{n_total:,} samples  |  {n_states} states  |  "
        f"Overall invalid: {n_invalid:,} ({100.*n_invalid/n_total:.1f}%)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Overview figure saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Scatter validity plot for IB-SparseAttention synthetic data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--synth-csv", default=str(SYNTH_CSV),
                   help="Path to synthetic CSV")
    p.add_argument("--mode", choices=["panel", "overview", "both"], default="panel",
                   help="Figure type to produce")
    p.add_argument("--states", nargs="+", default=None, metavar="STATE",
                   help='State codes to plot in panel mode, or "ALL"')
    p.add_argument("--n-top", type=int, default=4,
                   help="Default top-N states to show when --states not given")
    p.add_argument("--n-cols", type=int, default=2,
                   help="Columns in panel grid")
    p.add_argument("--color", choices=["validity", "lat_zone", "bird"],
                   default="validity",
                   help="Point coloring scheme")
    p.add_argument("--output-dir", default=str(OUTPUT_DIR),
                   help="Directory to save figures and stats CSV")
    return p.parse_args()


def main() -> None:
    args      = parse_args()
    out_dir   = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)
    synth_csv = Path(args.synth_csv)

    if not synth_csv.exists():
        log.error(f"Synthetic CSV not found: {synth_csv}")
        sys.exit(1)

    # ── Load data ─────────────────────────────────────────────────────────────
    df         = load_synthetic(synth_csv)
    geometries = load_all_geometries()

    # Filter to states that have both data and geometry
    available_geo   = set(geometries.keys())
    available_data  = set(df["state_code"].unique())
    available_both  = available_geo & available_data
    log.info(f"States with both data and geometry: {len(available_both)}")

    # ── Validity ──────────────────────────────────────────────────────────────
    log.info("Computing point-in-polygon validity (may take a moment)...")
    geo_for_pip = {k: geometries[k] for k in available_both}
    df = add_validity_column(df, geo_for_pip)

    # ── Stats ─────────────────────────────────────────────────────────────────
    stats_df   = compute_stats(df[df["state_code"].isin(available_both)])
    stats_path = out_dir / "scatter_stats_synthetic.csv"
    stats_df.to_csv(stats_path, index=False, encoding="utf-8")
    log.info(f"\nPer-state statistics:\n{stats_df.to_string(index=False)}")
    log.info(f"Stats saved: {stats_path}")

    # ── Panel figure ──────────────────────────────────────────────────────────
    if args.mode in ("panel", "both"):
        counts = df["state_code"].value_counts()

        if args.states is None:
            target = list(counts.index[:args.n_top])
            log.info(f"Panel: top-{args.n_top} states by count: {target}")
        elif len(args.states) == 1 and args.states[0].upper() == "ALL":
            target = [s for s in counts.index if s in available_both]
            log.info(f"Panel: all {len(target)} states with data+geometry")
        else:
            target = [s.upper() for s in args.states if s.upper() in available_both]
            log.info(f"Panel: user-specified states: {target}")

        if not target:
            log.error("No valid states to plot in panel mode.")
        else:
            panel_path = out_dir / "scatter_panel_synthetic.png"
            figure_panel(
                df=df, geometries=geometries, target_states=target,
                save_path=panel_path, n_cols=args.n_cols, color_by=args.color,
            )

    # ── Overview figure ───────────────────────────────────────────────────────
    if args.mode in ("overview", "both"):
        overview_path = out_dir / "scatter_overview_synthetic.png"
        figure_overview(
            df=df, geometries=geometries, stats_df=stats_df,
            save_path=overview_path, color_by=args.color,
        )

    log.info("\n=== Done ===")
    log.info(f"  Output directory: {out_dir}")
    if args.mode in ("panel", "both"):
        log.info(f"  scatter_panel_synthetic.png    -- per-state scatter grid")
    if args.mode in ("overview", "both"):
        log.info(f"  scatter_overview_synthetic.png -- full US choropleth + scatter")
    log.info(f"  scatter_stats_synthetic.csv    -- per-state statistics")


if __name__ == "__main__":
    main()

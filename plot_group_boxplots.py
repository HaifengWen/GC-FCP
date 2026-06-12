#!/usr/bin/env python3
"""
Plot group-level conformal prediction results as boxplots.

Default behavior:
  - reads raw_results.csv
  - filters to score == "raps"
  - plots one selected group_id
  - saves one two-panel figure: coverage and normalized set size

New large-figure behavior:
  - pass --big-all-groups to generate two big faceted figures:
      1. coverage for all groups
      2. normalized set size for all groups
  - in the big figures, the x/y axis titles are drawn once at the figure level.

Examples:
  # One group, two panels
  python plot_group_boxplots_v1.py \
      --csv raw_results.csv \
      --dataset imagenet \
      --score raps \
      --group-id 0

  # Two large figures: all-group coverage + all-group normalized set size
  python plot_group_boxplots_v1.py \
      --csv raw_results.csv \
      --dataset imagenet \
      --score raps \
      --big-all-groups

  # One figure per group, as in the previous version
  python plot_group_boxplots_v1.py \
      --csv raw_results.csv \
      --dataset imagenet \
      --score raps \
      --all-groups

Notes:
  - group-id is zero-based by default.
  - use --one-based-group-id if you want --group-id 1 to mean the first group.
  - use --methods to select a subset of method_key values.
"""

from __future__ import annotations

# These defaults avoid occasional over-threading issues in headless/batch jobs.
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd


# Order for the methods in raw_results.csv.
# Edit this if your CSV uses different method_key values.
METHOD_ORDER = [
    "cp",
    "fedcp",
    "mondrian",
    "fedcf",
    "iw_fcp",
    "gcfcp_delta_50",
    "gcfcp_delta_250",
    "gcfcp_delta_500",
]

# Human-readable labels. GC-FCP labels are also generated automatically by
# method_label() so that labels like gcfcp_delta_250 become
# GC-FCP\n($\delta=250$).
METHOD_LABELS = {
    "cp": "CP",
    "fedcp": "FedCP",
    "mondrian": "Mondrian-\nFedCP",
    "fedcf": "FedCF",
    "iw_fcp": "IW-FCP",
    "gcfcp_delta_50": "GC-FCP\n" + r"($\delta=50$)",
    "gcfcp_delta_250": "GC-FCP\n" + r"($\delta=250$)",
    "gcfcp_delta_500": "GC-FCP\n" + r"($\delta=500$)",
}

# ggplot-like outline colors. Extra methods cycle through this list.
METHOD_COLORS = [
    "#F8766D",
    "#A3A500",
    "#00BF7D",
    "#00B0F6",
    "#E76BF3",
    "#FF61C3",
    "#619CFF",
    "#00BA38",
]

PANEL_BG = "#FAFAFA"
GRID_MAJOR = "#E5E5E5"
GRID_MINOR = "#EFEFEF"
STRIP_BG = "#D9D9D9"
STRIP_EDGE = "#333333"


def parse_json_array(value: object, column: str) -> list[float]:
    """Parse a CSV cell containing a JSON list."""
    if pd.isna(value):
        raise ValueError(f"Column {column!r} contains a missing value where a JSON array is required.")
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse JSON array in column {column!r}: {value!r}") from exc
    if not isinstance(parsed, list):
        raise ValueError(f"Column {column!r} must contain a JSON list, got {type(parsed).__name__}.")
    return parsed


def method_label(method_key: str) -> str:
    """Return a neat plot label for a method_key."""
    method_key = str(method_key)
    if method_key.startswith("gcfcp_delta_"):
        delta = method_key.rsplit("_", 1)[-1]
        return "GC-FCP\n" + rf"($\delta={delta}$)"
    return METHOD_LABELS.get(method_key, method_key.replace("_", "-"))


def select_dataset(df: pd.DataFrame, dataset: str | None) -> tuple[pd.DataFrame, str | None]:
    """Filter to a dataset. If exactly one dataset exists, use it automatically."""
    if "dataset" not in df.columns:
        return df, None

    datasets = sorted(df["dataset"].dropna().astype(str).unique())
    if dataset is not None:
        out = df[df["dataset"].astype(str) == str(dataset)].copy()
        if out.empty:
            raise ValueError(f"No rows found for dataset={dataset!r}. Available datasets: {datasets}")
        return out, str(dataset)

    if len(datasets) == 1:
        return df[df["dataset"].astype(str) == datasets[0]].copy(), datasets[0]

    raise ValueError(
        "Multiple datasets are present. Please pass --dataset. "
        f"Available datasets: {', '.join(datasets)}"
    )


def filter_results(
    df: pd.DataFrame,
    *,
    score: str,
    dataset: str | None,
    alpha: float | None,
    methods: Sequence[str] | None,
) -> tuple[pd.DataFrame, str | None, float]:
    """Apply score/dataset/alpha/method filters and return the nominal coverage level."""
    required = {"score", "alpha", "method_key"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    df = df[df["score"].astype(str).str.lower() == score.lower()].copy()
    if df.empty:
        raise ValueError(f"No rows found for score={score!r}.")

    df, selected_dataset = select_dataset(df, dataset)

    if alpha is not None:
        df = df[df["alpha"].astype(float) == float(alpha)].copy()
        if df.empty:
            raise ValueError(f"No rows found for alpha={alpha} after score/dataset filtering.")

    if methods:
        keep = set(methods)
        df = df[df["method_key"].astype(str).isin(keep)].copy()
        if df.empty:
            raise ValueError(f"No rows found for requested methods: {sorted(keep)}")

    alphas = sorted(df["alpha"].dropna().astype(float).unique())
    if len(alphas) != 1:
        raise ValueError(f"Expected exactly one alpha after filtering, found {alphas}. Pass --alpha.")
    nominal_coverage = 1.0 - alphas[0]

    return df, selected_dataset, nominal_coverage


def available_group_count(df: pd.DataFrame) -> int:
    """Infer number of groups from group_cov_json."""
    if "group_cov_json" not in df.columns:
        raise ValueError("CSV is missing group_cov_json; group-level plotting is not possible.")
    lengths = df["group_cov_json"].map(lambda x: len(parse_json_array(x, "group_cov_json")))
    unique_lengths = sorted(lengths.unique())
    if len(unique_lengths) != 1:
        raise ValueError(f"Rows have inconsistent group counts: {unique_lengths}")
    return int(unique_lengths[0])


def make_group_frame(
    df: pd.DataFrame,
    *,
    group_id: int,
    size_column: str,
    size_scale: float,
) -> pd.DataFrame:
    """Build one row per Monte Carlo replicate/method for the selected group."""
    size_json_col = {
        "norm": "group_norm_avg_set_size_json",
        "raw": "group_avg_set_size_json",
    }[size_column]
    required = {"method_key", "method", "group_cov_json", size_json_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    records: list[dict[str, object]] = []
    for row_idx, row in df.iterrows():
        cov = parse_json_array(row["group_cov_json"], "group_cov_json")
        size = parse_json_array(row[size_json_col], size_json_col)
        group_sizes = parse_json_array(row["group_size_json"], "group_size_json") if "group_size_json" in df.columns else None

        if group_id < 0 or group_id >= len(cov):
            raise ValueError(f"group_id={group_id} is out of range for row {row_idx}; valid range is 0..{len(cov)-1}.")
        if len(size) != len(cov):
            raise ValueError(f"Row {row_idx} has mismatched coverage and size group counts.")

        records.append(
            {
                "seed": row.get("seed", math.nan),
                "mc": row.get("mc", math.nan),
                "method_key": str(row["method_key"]),
                "method": str(row["method"]),
                "coverage": float(cov[group_id]),
                "set_size": float(size[group_id]) * size_scale,
                "group_id": int(group_id),
                "group_size": int(group_sizes[group_id]) if group_sizes is not None else math.nan,
            }
        )

    out = pd.DataFrame.from_records(records)
    if out.empty:
        raise ValueError("No data were available to plot after filtering.")
    return out


def make_all_groups_frame(
    df: pd.DataFrame,
    *,
    n_groups: int,
    size_column: str,
    size_scale: float,
) -> pd.DataFrame:
    """Build one long DataFrame containing all groups."""
    frames = [
        make_group_frame(df, group_id=gid, size_column=size_column, size_scale=size_scale)
        for gid in range(n_groups)
    ]
    return pd.concat(frames, ignore_index=True)


def sorted_methods(present_methods: Iterable[str]) -> list[str]:
    """Order known methods first; append unknown methods alphabetically."""
    present = list(dict.fromkeys(str(m) for m in present_methods))
    known = [m for m in METHOD_ORDER if m in present]
    unknown = sorted(m for m in present if m not in METHOD_ORDER)
    return known + unknown


def color_map(methods: Sequence[str]) -> dict[str, str]:
    return {m: METHOD_COLORS[i % len(METHOD_COLORS)] for i, m in enumerate(methods)}


def style_axis(ax: plt.Axes) -> None:
    """Apply a light ggplot-like panel style."""
    ax.set_facecolor(PANEL_BG)
    ax.grid(True, axis="y", which="major", color=GRID_MAJOR, linewidth=1.2)
    ax.grid(True, axis="x", which="major", color=GRID_MAJOR, linewidth=1.2)
    ax.grid(True, axis="y", which="minor", color=GRID_MINOR, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    for spine in ax.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(1.0)
    ax.tick_params(axis="both", labelsize=10, width=1.0, colors="#333333")


def add_strip(ax: plt.Axes, label: str, fontsize: float = 12) -> None:
    """Add a grey facet strip above an axis, matching the reference style."""
    ax.text(
        0.5,
        1.03,
        label,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        color="#111111",
        bbox={
            "boxstyle": "square,pad=0.35",
            "facecolor": STRIP_BG,
            "edgecolor": STRIP_EDGE,
            "linewidth": 1.0,
        },
    )


def draw_colored_boxplot(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    value_col: str,
    methods: Sequence[str],
    method_colors: dict[str, str],
    ylabel: str | None = None,
    show_x_ticklabels: bool = True,
    tick_labelsize: float = 10,
    tick_rotation: float = 35,
) -> None:
    """Draw white-filled boxplots with method-colored outlines and fliers."""
    vals = [data.loc[data["method_key"] == m, value_col].dropna().to_numpy() for m in methods]
    positions = list(range(1, len(methods) + 1))

    bp = ax.boxplot(
        vals,
        positions=positions,
        widths=0.58,
        patch_artist=True,
        showfliers=True,
        showmeans=True,
        whis=1.5,
        boxprops={"linewidth": 1.8},
        medianprops={"linewidth": 1.8, "color": "#333333"},
        whiskerprops={"linewidth": 1.5},
        capprops={"linewidth": 1.5},
        flierprops={"marker": "o", "markersize": 4.7, "markeredgewidth": 0.0},
        meanprops={
            "marker": "*",
            "markerfacecolor": "red",
            "markeredgecolor": "red",
            "markersize": 9.0,
            "linestyle": "none",
        },
    )

    for i, method in enumerate(methods):
        color = method_colors[method]
        bp["boxes"][i].set(facecolor="white", edgecolor=color, linewidth=1.8)
        bp["medians"][i].set(color=color, linewidth=1.8)
        # Red stars mark the mean values for each method.
        if "means" in bp and i < len(bp["means"]):
            bp["means"][i].set(marker="*", markerfacecolor="red", markeredgecolor="red", markersize=9.0)
        bp["fliers"][i].set(markerfacecolor=color, markeredgecolor=color, alpha=0.95)
        for obj in bp["whiskers"][2 * i : 2 * i + 2]:
            obj.set(color=color, linewidth=1.5)
        for obj in bp["caps"][2 * i : 2 * i + 2]:
            obj.set(color=color, linewidth=1.5)

    labels = [method_label(m) for m in methods]
    ax.set_xticks(positions)
    if show_x_ticklabels:
        ax.set_xticklabels(labels, rotation=tick_rotation, ha="right", fontsize=tick_labelsize)
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=13)
    style_axis(ax)


def coverage_limits(values: pd.Series, nominal: float) -> tuple[float, float]:
    """Choose compact coverage limits while respecting [0, 1]."""
    ymin = min(float(values.min()), nominal) - 0.03
    ymax = max(float(values.max()), nominal) + 0.03
    ymin = max(0.0, ymin)
    ymax = min(1.0, ymax)
    if ymax - ymin < 0.10:
        mid = 0.5 * (ymin + ymax)
        ymin = max(0.0, mid - 0.05)
        ymax = min(1.0, mid + 0.05)
    return ymin + 0.3, ymax


def set_coverage_limits(ax: plt.Axes, values: pd.Series, nominal: float) -> None:
    ymin, ymax = coverage_limits(values, nominal)
    ax.set_ylim(ymin, ymax)


def nice_upper(values: pd.Series, pad_fraction: float = 0.08) -> float:
    """Return a lightly padded upper y-limit for positive quantities."""
    vmax = float(values.max())
    if not math.isfinite(vmax) or vmax <= 0:
        return 1.0
    return vmax * (1.0 + pad_fraction)


def plot_one_group(
    group_df: pd.DataFrame,
    *,
    group_id: int,
    display_group_id: int,
    score: str,
    dataset: str | None,
    nominal_coverage: float,
    size_label: str,
    out_path: Path,
    dpi: int,
) -> None:
    methods = sorted_methods(group_df["method_key"].unique())
    method_colors = color_map(methods)

    width = max(9.0, 1.05 * len(methods) + 3.0)
    fig, axes = plt.subplots(1, 2, figsize=(width, 4.6), constrained_layout=True)

    draw_colored_boxplot(
        axes[0],
        group_df,
        value_col="coverage",
        methods=methods,
        method_colors=method_colors,
        ylabel="Coverage",
        tick_labelsize=10.5,
    )
    axes[0].axhline(nominal_coverage, color="#222222", linestyle=(0, (4, 4)), linewidth=1.2)
    set_coverage_limits(axes[0], group_df["coverage"], nominal_coverage)
    add_strip(axes[0], "Coverage", fontsize=13)

    draw_colored_boxplot(
        axes[1],
        group_df,
        value_col="set_size",
        methods=methods,
        method_colors=method_colors,
        ylabel=size_label,
        tick_labelsize=10.5,
    )
    axes[1].set_ylim(bottom=0)
    add_strip(axes[1], size_label, fontsize=13)

    for ax in axes:
        ax.set_xlabel("Method", fontsize=13)

    group_sizes = sorted(group_df["group_size"].dropna().unique())
    group_size_text = f", n={int(group_sizes[0])}" if len(group_sizes) == 1 else ""
    dataset_text = f"{dataset}, " if dataset else ""
    fig.suptitle(
        f"{dataset_text}{score.upper()} results for group {display_group_id}{group_size_text}",
        fontsize=14,
        y=1.04,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_big_metric(
    all_groups_df: pd.DataFrame,
    *,
    metric: str,
    metric_label: str,
    n_groups: int,
    score: str,
    dataset: str | None,
    nominal_coverage: float,
    out_path: Path,
    dpi: int,
    ncols: int,
    one_based_group_labels: bool,
    show_all_xticklabels: bool,
    share_y: bool,
) -> None:
    """Plot one large faceted figure for either coverage or set_size."""
    methods = sorted_methods(all_groups_df["method_key"].unique())
    method_colors = color_map(methods)

    ncols = max(1, min(int(ncols), n_groups))
    nrows = int(math.ceil(n_groups / ncols))

    # Wider panels are useful because several method labels contain line breaks.
    panel_width = max(3.1, 0.42 * len(methods) + 1.2)
    panel_height = 2.65 if metric == "coverage" else 2.75
    fig_width = panel_width * ncols
    fig_height = panel_height * nrows + 0.85

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_width, fig_height),
        constrained_layout=True,
        squeeze=False,
        sharey=share_y,
    )
    axes_flat = list(axes.flat)

    if metric == "coverage" and share_y:
        ylimits = coverage_limits(all_groups_df["coverage"], nominal_coverage)
    elif metric == "set_size" and share_y:
        ylimits = (0.0, nice_upper(all_groups_df["set_size"]))
    else:
        ylimits = None

    for gid in range(n_groups):
        ax = axes_flat[gid]
        group_df = all_groups_df[all_groups_df["group_id"] == gid]
        row = gid // ncols
        is_bottom_row = row == nrows - 1
        display_gid = gid + 1 if one_based_group_labels else gid
        group_sizes = sorted(group_df["group_size"].dropna().unique())
        group_size_text = f" (n={int(group_sizes[0])})" if len(group_sizes) == 1 else ""

        draw_colored_boxplot(
            ax,
            group_df,
            value_col=metric,
            methods=methods,
            method_colors=method_colors,
            ylabel=None,
            show_x_ticklabels=show_all_xticklabels or is_bottom_row,
            tick_labelsize=8.0,
            tick_rotation=35,
        )
        add_strip(ax, f"Group {display_gid}{group_size_text}", fontsize=10.5)

        if metric == "coverage":
            ax.axhline(nominal_coverage, color="#222222", linestyle=(0, (4, 4)), linewidth=1.05)
            if ylimits is None:
                set_coverage_limits(ax, group_df["coverage"], nominal_coverage)
            else:
                ax.set_ylim(*ylimits)
        else:
            if ylimits is None:
                ax.set_ylim(0.0, nice_upper(group_df["set_size"]))
            else:
                ax.set_ylim(*ylimits)

        # Axis titles should appear only once in the big figures.
        ax.set_xlabel("")
        ax.set_ylabel("")

    for ax in axes_flat[n_groups:]:
        ax.set_visible(False)

    dataset_text = f"{dataset}, " if dataset else ""
    fig.suptitle(f"{dataset_text}{score.upper()} {metric_label} across all groups", fontsize=15)
    fig.supxlabel("Method", fontsize=14)
    fig.supylabel(metric_label, fontsize=14)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def default_output_path(csv_path: Path, dataset: str | None, score: str, group_id: int, suffix: str) -> Path:
    name_parts = []
    if dataset:
        name_parts.append(dataset)
    name_parts.extend([score.lower(), f"group{group_id}"])
    return csv_path.parent / "figures" / ("_".join(name_parts) + suffix)


def default_big_output_path(
    csv_path: Path,
    dataset: str | None,
    score: str,
    metric_stub: str,
    suffix: str,
    out_dir: Path | None,
) -> Path:
    name_parts = []
    if dataset:
        name_parts.append(dataset)
    name_parts.extend([score.lower(), "all_groups", metric_stub])
    base_dir = out_dir if out_dir is not None else csv_path.parent / "figures"
    return base_dir / ("_".join(name_parts) + suffix)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot group-level coverage/size boxplot figures.")
    parser.add_argument("--csv", default="raw_results.csv", help="Path to the results CSV.")
    parser.add_argument("--dataset", default=None, help="Dataset to plot. Required if CSV contains multiple datasets.")
    parser.add_argument("--score", default="raps", help="Score to plot; default: raps.")
    parser.add_argument("--alpha", type=float, default=None, help="Alpha to plot. Required if multiple alpha values remain after filtering.")
    parser.add_argument("--group-id", type=int, default=0, help="Group id to plot. Zero-based unless --one-based-group-id is passed.")
    parser.add_argument("--one-based-group-id", action="store_true", help="Interpret --group-id as one-based and show group labels as one-based.")
    parser.add_argument("--all-groups", action="store_true", help="Create one two-panel figure for each group.")
    parser.add_argument("--big-all-groups", action="store_true", help="Create two big figures: all-group coverage and all-group set size.")
    parser.add_argument("--big-ncols", type=int, default=5, help="Number of columns in each big all-group figure. Default: 5.")
    parser.add_argument("--big-show-all-xticklabels", action="store_true", help="Show method tick labels in every panel of big figures. Default: bottom row only.")
    parser.add_argument("--free-y", action="store_true", help="Use separate y-limits for each panel in the big figures. Default: shared y-limits.")
    parser.add_argument("--methods", nargs="*", default=None, help="Optional method_key values to include.")
    parser.add_argument("--size-column", choices=["norm", "raw"], default="norm", help="Plot normalized or raw average set size. Default: norm.")
    parser.add_argument("--size-scale", type=float, default=1.0, help="Multiply the chosen size values by this number before plotting.")
    parser.add_argument("--out", default=None, help="Output image path for one group. Ignored when --all-groups or --big-all-groups is used.")
    parser.add_argument("--out-dir", default=None, help="Output directory for --all-groups and --big-all-groups. Default: <csv_dir>/figures.")
    parser.add_argument("--big-coverage-out", default=None, help="Optional output path for the big coverage figure.")
    parser.add_argument("--big-size-out", default=None, help="Optional output path for the big set-size figure.")
    parser.add_argument("--suffix", default=".png", help="Output suffix/format, e.g. .png or .pdf. Default: .png.")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for raster output. Default: 300.")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)
    df, dataset, nominal = filter_results(
        df,
        score=args.score,
        dataset=args.dataset,
        alpha=args.alpha,
        methods=args.methods,
    )

    n_groups = available_group_count(df)
    requested_group = args.group_id - 1 if args.one_based_group_id else args.group_id
    out_dir = Path(args.out_dir) if args.out_dir else None

    size_base_label = "Normalized set size" if args.size_column == "norm" else "Set size"
    size_label = size_base_label if args.size_scale == 1 else f"{size_base_label} x {args.size_scale:g}"
    size_metric_stub = "norm_set_size" if args.size_column == "norm" else "set_size"

    if args.big_all_groups:
        all_groups_df = make_all_groups_frame(
            df,
            n_groups=n_groups,
            size_column=args.size_column,
            size_scale=args.size_scale,
        )
        coverage_out = Path(args.big_coverage_out) if args.big_coverage_out else default_big_output_path(
            csv_path, dataset, args.score, "coverage", args.suffix, out_dir
        )
        size_out = Path(args.big_size_out) if args.big_size_out else default_big_output_path(
            csv_path, dataset, args.score, size_metric_stub, args.suffix, out_dir
        )

        plot_big_metric(
            all_groups_df,
            metric="coverage",
            metric_label="Coverage",
            n_groups=n_groups,
            score=args.score,
            dataset=dataset,
            nominal_coverage=nominal,
            out_path=coverage_out,
            dpi=args.dpi,
            ncols=args.big_ncols,
            one_based_group_labels=args.one_based_group_id,
            show_all_xticklabels=args.big_show_all_xticklabels,
            share_y=not args.free_y,
        )
        print(f"Saved {coverage_out}")

        plot_big_metric(
            all_groups_df,
            metric="set_size",
            metric_label=size_label,
            n_groups=n_groups,
            score=args.score,
            dataset=dataset,
            nominal_coverage=nominal,
            out_path=size_out,
            dpi=args.dpi,
            ncols=args.big_ncols,
            one_based_group_labels=args.one_based_group_id,
            show_all_xticklabels=args.big_show_all_xticklabels,
            share_y=not args.free_y,
        )
        print(f"Saved {size_out}")

    if args.all_groups:
        output_dir = out_dir if out_dir is not None else csv_path.parent / "figures"
        group_ids = list(range(n_groups))
    elif not args.big_all_groups:
        if requested_group < 0 or requested_group >= n_groups:
            raise ValueError(f"group_id={requested_group} is out of range; valid zero-based ids are 0..{n_groups - 1}.")
        output_dir = None
        group_ids = [requested_group]
    else:
        group_ids = []
        output_dir = None

    for gid in group_ids:
        group_df = make_group_frame(
            df,
            group_id=gid,
            size_column=args.size_column,
            size_scale=args.size_scale,
        )
        if args.all_groups:
            out_path = output_dir / default_output_path(csv_path, dataset, args.score, gid, args.suffix).name
        elif args.out:
            out_path = Path(args.out)
        else:
            out_path = default_output_path(csv_path, dataset, args.score, gid, args.suffix)

        display_gid = gid + 1 if args.one_based_group_id else gid
        plot_one_group(
            group_df,
            group_id=gid,
            display_group_id=display_gid,
            score=args.score,
            dataset=dataset,
            nominal_coverage=nominal,
            size_label=size_label,
            out_path=out_path,
            dpi=args.dpi,
        )
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

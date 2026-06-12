#!/usr/bin/env python3
"""
No-exact-baseline ablation runner for the GC-FCP rebuttal experiments.

This script intentionally contains only orchestration, recording, table generation,
and plotting logic. It reuses the data generation, GC-FCP implementation, evaluation,
and summary utilities from `realdata_rebuttal.py`.

Ablations implemented
---------------------
A. Group design ablation:
   Run GC-FCP under several matched calibration/evaluation group families.
   The default designs are
     * semantic: 10 ImageNet semantic meta-groups;
     * ambiguity: ambiguity bins only;
     * semantic_ambiguity: union of semantic groups and ambiguity bins.

B. Compression ablation:
   Run GC-FCP while sweeping manuscript T-Digest delta values.

C. Federated aggregation ablation:
   Run GC-FCP while sweeping the number of clients K.

The `tables` and `plots` subcommands operate only on saved raw numerical results,
so tables/figures can be regenerated without rerunning experiments.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Matplotlib is only needed for the plots command. Use a non-interactive backend.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make imports robust when this script is copied next to realdata_rebuttal.py.
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from realdata_rebuttal import (  # noqa: E402
    ExperimentData,
    fmt_delta,
    fmt_mean_se,
    make_real_data,
    mean_se,
    normalize_pi,
    parse_csv_list,
    pooled_group_summary,
    run_methods,
)


# -------------------------
# Generic helpers
# -------------------------


def parse_str_list(s: str) -> List[str]:
    return [x.strip() for x in str(s or "").split(",") if x.strip()]


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def base_namespace(args: argparse.Namespace, K: int, scores: str, deltas: str) -> argparse.Namespace:
    """Build a minimal namespace matching the arguments needed by make_real_data/run_methods."""
    return argparse.Namespace(
        dataset=str(args.dataset),
        scores=str(scores),
        clients=int(K),
        times=int(args.times),
        alpha=float(args.alpha),
        n_cal=int(args.n_cal),
        n_test=int(args.n_test),
        pi="",
        base_seed=int(args.base_seed),
        out_dir=str(args.out_dir),
        quiet=bool(args.quiet),
        n_jobs=int(args.n_jobs),
        methods="gcfcp",
        include_exact=False,
        deltas=str(deltas),
        tdigest_K=int(args.tdigest_K),
        group_rule=str(args.group_rule),
        lp_max_iter=int(args.lp_max_iter),
        synthetic_train=1000,
        poly_degree=4,
        no_synthetic_label_shift=False,
        group_mode=str(args.group_mode),
        x_feature=str(args.x_feature),
        client_split=str(args.client_split),
        dirichlet_beta=float(args.dirichlet_beta),
        min_client_count=int(args.min_client_count),
        top_r=int(args.top_r),
        batch_size=int(args.batch_size),
        data_root=str(args.data_root),
        imagenet_val_dir=str(args.imagenet_val_dir),
        imagenet_class_index_json=str(args.imagenet_class_index_json),
        imagenet_meta_map=str(args.imagenet_meta_map),
        imagenet_model=str(args.imagenet_model),
        imagenet_cache=str(args.imagenet_cache),
        ambiguity_score=str(args.ambiguity_score),
        ambiguity_bins=int(args.ambiguity_bins),
        ambiguity_binning=str(args.ambiguity_binning),
        raps_lambda=float(args.raps_lambda),
        raps_k_reg=int(args.raps_k_reg),
    )


def make_base_real_data(
    args: argparse.Namespace,
    seed: int,
    score: str,
    K: int,
    group_mode: Optional[str] = None,
    top_r: Optional[int] = None,
) -> ExperimentData:
    pi = normalize_pi(None, int(K))
    return make_real_data(
        seed=int(seed),
        dataset=str(args.dataset),
        score=str(score),
        alpha=float(args.alpha),
        K=int(K),
        pi=pi,
        n_cal=int(args.n_cal),
        n_test=int(args.n_test),
        group_mode=str(group_mode or args.group_mode),
        top_r=int(args.top_r if top_r is None else top_r),
        raps_lambda=float(args.raps_lambda),
        raps_k_reg=int(args.raps_k_reg),
        batch_size=int(args.batch_size),
        x_feature=str(args.x_feature),
        client_split=str(args.client_split),
        dirichlet_beta=float(args.dirichlet_beta),
        min_client_count=int(args.min_client_count),
        data_root=str(args.data_root),
        imagenet_val_dir=str(args.imagenet_val_dir),
        imagenet_class_index_json=str(args.imagenet_class_index_json),
        imagenet_meta_map=str(args.imagenet_meta_map),
        imagenet_model=str(args.imagenet_model),
        imagenet_cache=str(args.imagenet_cache),
        ambiguity_score=str(args.ambiguity_score),
        ambiguity_bins=int(args.ambiguity_bins),
        ambiguity_binning=str(args.ambiguity_binning),
    )


# -------------------------
# Group-design construction
# -------------------------


def _subset_columns_data(
    data: ExperimentData,
    cols: Sequence[int],
    names: Sequence[str],
    group_design: str,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> ExperimentData:
    cols = list(map(int, cols))
    metadata = dict(data.metadata)
    metadata.update(
        {
            "ablation_group_design": str(group_design),
            "ablation_group_columns": cols,
            "ablation_group_names": list(names),
        }
    )
    if extra_meta:
        metadata.update(extra_meta)
    return replace(
        data,
        group_names=list(names),
        client_phi=[np.asarray(phi)[:, cols].astype(int) for phi in data.client_phi],
        test_phi=np.asarray(data.test_phi)[:, cols].astype(int),
        metadata=metadata,
    )


def slice_semantic_ambiguity_design(base: ExperimentData, design: str, ambiguity_bins: int) -> ExperimentData:
    """Slice a full semantic+ambiguity ExperimentData object into a matched design."""
    design = str(design).lower()
    B = int(ambiguity_bins)
    n_sem = 10
    if base.n_groups < n_sem + B:
        raise RuntimeError(f"Expected at least {n_sem + B} semantic+ambiguity groups; got {base.n_groups}.")
    if design == "semantic":
        cols = list(range(n_sem))
        names = list(base.group_names[:n_sem])
        return _subset_columns_data(base, cols, names, "semantic")
    if design == "ambiguity":
        cols = list(range(n_sem, n_sem + B))
        names = list(base.group_names[n_sem : n_sem + B])
        return _subset_columns_data(base, cols, names, "ambiguity")
    if design in {"semantic_ambiguity", "semamb"}:
        cols = list(range(n_sem + B))
        names = list(base.group_names[: n_sem + B])
        return _subset_columns_data(base, cols, names, "semantic_ambiguity")
    raise ValueError(f"Design {design} is not derived from semantic_ambiguity.")


def make_group_design_data(
    args: argparse.Namespace,
    seed: int,
    score: str,
    K: int,
    design: str,
) -> ExperimentData:
    """Return ExperimentData whose calibration and evaluation groups are matched."""
    design = str(design).lower()
    B = int(args.ambiguity_bins)

    if design in {"semantic", "ambiguity", "semantic_ambiguity", "semamb"}:
        base = make_base_real_data(
            args,
            seed=seed,
            score=score,
            K=K,
            group_mode="imagenet_semantic_ambiguity",
            top_r=1,
        )
        return slice_semantic_ambiguity_design(base, design, B)

    if design in {"semantic_top2", "meta_top2"}:
        data = make_base_real_data(
            args,
            seed=seed,
            score=score,
            K=K,
            group_mode="imagenet_meta_topr",
            top_r=2,
        )
        metadata = dict(data.metadata)
        metadata["ablation_group_design"] = "semantic_top2"
        return replace(data, metadata=metadata)

    if design in {"semantic_top1", "meta_top1"}:
        data = make_base_real_data(
            args,
            seed=seed,
            score=score,
            K=K,
            group_mode="imagenet_meta_topr",
            top_r=1,
        )
        metadata = dict(data.metadata)
        metadata["ablation_group_design"] = "semantic_top1"
        return replace(data, metadata=metadata)

    raise ValueError(
        f"Unknown group design '{design}'. Supported designs: "
        "semantic, ambiguity, semantic_ambiguity, semantic_top1, semantic_top2."
    )


# -------------------------
# Running experiments
# -------------------------


def record_dict(
    rec: Any,
    ablation: str,
    setting: str,
    group_design: str,
    clients: int,
    delta: float,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    d = asdict(rec)
    d.update(
        {
            "ablation": str(ablation),
            "setting": str(setting),
            "group_design": str(group_design),
            "clients": int(clients),
            "delta": float(delta),
        }
    )
    if extra:
        for k, v in extra.items():
            if isinstance(v, (dict, list, tuple)):
                d[k] = json.dumps(v)
            else:
                d[k] = v
    return d


def run_gcfcp_records(
    data: ExperimentData,
    seed: int,
    mc: int,
    deltas: Sequence[float],
    args: argparse.Namespace,
) -> List[Any]:
    rng = np.random.default_rng(int(seed) + 1000003)
    return run_methods(
        data=data,
        seed=int(seed),
        mc=int(mc),
        method_keys=["gcfcp"],
        deltas=[float(d) for d in deltas],
        tdigest_K=int(args.tdigest_K),
        group_rule=str(args.group_rule),
        lp_max_iter=int(args.lp_max_iter),
        rng=rng,
        n_jobs=int(args.n_jobs),
    )


def run_group_ablation(args: argparse.Namespace, rows: List[Dict[str, Any]], raw_path: Path) -> None:
    scores = parse_str_list(args.scores)
    designs = parse_str_list(args.group_designs)
    delta = float(args.group_delta)
    K = int(args.clients)

    semamb_designs = {"semantic", "ambiguity", "semantic_ambiguity", "semamb"}

    for mc in range(int(args.times)):
        seed = int(args.base_seed) + mc
        for score in scores:
            if not args.quiet:
                print(f"[group] mc={mc+1}/{args.times} score={score} seed={seed}")

            # Reuse one full semantic+ambiguity data object for all designs that
            # are column subsets of that family. This keeps the data split, client
            # split, scores, and base predictions identical across group designs.
            base_semamb = None
            if any(str(d).lower() in semamb_designs for d in designs):
                base_semamb = make_base_real_data(
                    args,
                    seed=seed,
                    score=score,
                    K=K,
                    group_mode="imagenet_semantic_ambiguity",
                    top_r=1,
                )

            for design in designs:
                design_l = str(design).lower()
                if design_l in semamb_designs:
                    assert base_semamb is not None
                    data = slice_semantic_ambiguity_design(base_semamb, design_l, int(args.ambiguity_bins))
                else:
                    data = make_group_design_data(args, seed, score, K, design_l)
                recs = run_gcfcp_records(data, seed, mc, [delta], args)
                for rec in recs:
                    rows.append(
                        record_dict(
                            rec,
                            ablation="group",
                            setting=design_l,
                            group_design=design_l,
                            clients=K,
                            delta=delta,
                            extra={"n_groups_design": data.n_groups},
                        )
                    )
                pd.DataFrame(rows).to_csv(raw_path, index=False)


def run_delta_ablation(args: argparse.Namespace, rows: List[Dict[str, Any]], raw_path: Path) -> None:
    scores = parse_str_list(args.scores)
    deltas = parse_csv_list(args.delta_values, float)
    K = int(args.clients)
    design = str(args.delta_group_design)

    for mc in range(int(args.times)):
        seed = int(args.base_seed) + mc
        for score in scores:
            if not args.quiet:
                print(f"[delta] mc={mc+1}/{args.times} score={score} seed={seed} deltas={deltas}")
            data = make_group_design_data(args, seed, score, K, design)
            recs = run_gcfcp_records(data, seed, mc, deltas, args)
            for rec in recs:
                # Extract the actual delta from the method key/details.
                delta = float(json.loads(rec.details_json).get("paper_delta", np.nan))
                rows.append(
                    record_dict(
                        rec,
                        ablation="delta",
                        setting=fmt_delta(delta),
                        group_design=design,
                        clients=K,
                        delta=delta,
                        extra={"n_groups_design": data.n_groups},
                    )
                )
            pd.DataFrame(rows).to_csv(raw_path, index=False)


def run_k_ablation(args: argparse.Namespace, rows: List[Dict[str, Any]], raw_path: Path) -> None:
    scores = parse_str_list(args.scores)
    k_values = [int(k) for k in parse_csv_list(args.k_values, int)]
    delta = float(args.k_delta)
    design = str(args.k_group_design)

    for mc in range(int(args.times)):
        seed = int(args.base_seed) + mc
        for score in scores:
            for K in k_values:
                if not args.quiet:
                    print(f"[k] mc={mc+1}/{args.times} score={score} seed={seed} K={K}")
                data = make_group_design_data(args, seed, score, K, design)
                recs = run_gcfcp_records(data, seed, mc, [delta], args)
                for rec in recs:
                    rows.append(
                        record_dict(
                            rec,
                            ablation="k",
                            setting=str(K),
                            group_design=design,
                            clients=K,
                            delta=delta,
                            extra={"n_groups_design": data.n_groups},
                        )
                    )
                pd.DataFrame(rows).to_csv(raw_path, index=False)


def run_selected_ablations(args: argparse.Namespace) -> Path:
    out_root = ensure_dir(args.out_dir)
    run_dir = ensure_dir(out_root / f"gcfcp_ablation_{now_stamp()}")
    raw_path = run_dir / "ablation_raw_results.csv"
    config_path = run_dir / "config.json"

    args_dict = vars(args).copy()
    args_dict["script"] = Path(__file__).name
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(args_dict, f, indent=2, sort_keys=True)

    selected = str(args.ablation).lower()
    rows: List[Dict[str, Any]] = []

    if selected in {"group", "all"}:
        run_group_ablation(args, rows, raw_path)
    if selected in {"delta", "all"}:
        run_delta_ablation(args, rows, raw_path)
    if selected in {"k", "all"}:
        run_k_ablation(args, rows, raw_path)

    if not raw_path.exists():
        pd.DataFrame(rows).to_csv(raw_path, index=False)

    make_tables(raw_path, run_dir)
    make_plots(raw_path, run_dir, coverage_metric=str(args.coverage_metric))

    if not args.quiet:
        print(f"Saved raw results: {raw_path}")
        print(f"Saved output directory: {run_dir}")
    return run_dir


# -------------------------
# Table generation
# -------------------------


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _mean_se_cell(sub: pd.DataFrame, col: str, digits: int = 3, integer: bool = False) -> str:
    if col not in sub.columns:
        return "N/A"
    m, se = mean_se(sub[col])
    return fmt_mean_se(m, se, digits=digits, integer=integer)


def summarize_subset(sub: pd.DataFrame) -> Dict[str, str]:
    alpha = 0.1
    if "alpha" in sub.columns:
        vals = _numeric(sub["alpha"]).dropna()
        if len(vals):
            alpha = float(vals.iloc[0])

    # Keep worst-group coverage as the pooled estimate across Monte Carlo
    # repetitions: min_g sum_m covered_{m,g} / sum_m n_{m,g}.
    pooled_worst, _, pooled_gap, _, _ = pooled_group_summary(sub, alpha)

    # Report average group coverage as the Monte Carlo mean of the per-run
    # unweighted average across groups: (1/M) sum_m (1/|G|) sum_g cov_{m,g}.
    # The raw records store this per-run quantity in `avg_group_cov`, so the
    # standard error is computed across Monte Carlo repetitions.
    if "avg_group_cov" in sub.columns:
        avg_group_cell = _mean_se_cell(sub, "avg_group_cov", digits=3)
    else:
        vals = []
        for _, row in sub.iterrows():
            try:
                cov = np.asarray(json.loads(row.get("group_cov_json", "[]")), dtype=float)
            except Exception:
                cov = np.array([], dtype=float)
            cov = cov[np.isfinite(cov)]
            if cov.size > 0:
                vals.append(float(np.mean(cov)))
        m, se = mean_se(pd.Series(vals, dtype=float)) if vals else (float("nan"), float("nan"))
        avg_group_cell = fmt_mean_se(m, se, digits=3)

    return {
        "Marg. cov.": _mean_se_cell(sub, "marg_cov", digits=3),
        "Worst-group cov.": fmt_mean_se(pooled_worst, 0.0, digits=3),
        "Avg group cov.": avg_group_cell,
        "Max group gap": fmt_mean_se(pooled_gap, 0.0, digits=3),
        "Avg set size": _mean_se_cell(sub, "avg_set_size", digits=3),
        "Norm set size": _mean_se_cell(sub, "norm_avg_set_size", digits=4),
        "Atoms": _mean_se_cell(sub, "atoms", digits=1, integer=True),
        "Coreset": _mean_se_cell(sub, "coreset", digits=1, integer=True),
        "Comm.": _mean_se_cell(sub, "comm", digits=1, integer=True),
        "Runtime (s)": _mean_se_cell(sub, "runtime_sec", digits=3),
        "Comp. speedup": _mean_se_cell(sub, "comp_speedup", digits=2),
    }
def make_summary_table(df: pd.DataFrame, ablation: str) -> pd.DataFrame:
    work = df[df["ablation"] == ablation].copy()
    if work.empty:
        return pd.DataFrame()

    if ablation == "group":
        group_cols = ["score", "group_design", "delta", "clients"]
        label_cols = ["Score", "Group design", "Delta", "K"]
    elif ablation == "delta":
        group_cols = ["score", "delta", "group_design", "clients"]
        label_cols = ["Score", "Delta", "Group design", "K"]
    elif ablation == "k":
        group_cols = ["score", "clients", "group_design", "delta"]
        label_cols = ["Score", "K", "Group design", "Delta"]
    else:
        raise ValueError(f"Unknown ablation {ablation}")

    rows: List[Dict[str, str]] = []
    for keys, sub in work.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: Dict[str, str] = {}
        for label, value in zip(label_cols, keys):
            if label == "Score":
                row[label] = str(value).upper()
            elif label == "Delta":
                row[label] = fmt_delta(float(value))
            else:
                row[label] = str(value)
        row.update(summarize_subset(sub))
        rows.append(row)

    cols = label_cols + [
        "Marg. cov.", "Worst-group cov.", "Avg group cov.", "Max group gap",
        "Avg set size", "Norm set size", "Atoms", "Coreset", "Comm.",
        "Runtime (s)", "Comp. speedup",
    ]
    return pd.DataFrame(rows)[cols]


def make_tables(results_csv: Path | str, out_dir: Path | str) -> Dict[str, pd.DataFrame]:
    out = ensure_dir(out_dir)
    df = pd.read_csv(results_csv)
    tables: Dict[str, pd.DataFrame] = {}
    for ablation in ["group", "delta", "k"]:
        tab = make_summary_table(df, ablation)
        if not tab.empty:
            tables[ablation] = tab
            tab.to_csv(out / f"table_{ablation}.csv", index=False)
            with open(out / f"table_{ablation}.md", "w", encoding="utf-8") as f:
                f.write(tab.to_markdown(index=False))
                f.write("\n")
    return tables


# -------------------------
# Plot generation
# -------------------------


def curve_summary(df: pd.DataFrame, ablation: str, x_col: str, y_col: str) -> pd.DataFrame:
    work = df[df["ablation"] == ablation].copy()
    if work.empty:
        return pd.DataFrame(columns=["score", x_col, "mean", "se"])
    rows: List[Dict[str, Any]] = []
    for (score, x), sub in work.groupby(["score", x_col], sort=True):
        vals = pd.to_numeric(sub[y_col], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        rows.append(
            {
                "score": str(score),
                x_col: float(x),
                "mean": float(np.mean(vals)),
                "se": float(np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _plot_curve(
    summary: pd.DataFrame,
    x_col: str,
    y_label: str,
    title: str,
    out_path: Path,
    target: Optional[float] = None,
    log_x: bool = False,
) -> None:
    if summary.empty:
        return
    fig = plt.figure(figsize=(6.2, 4.2))
    ax = fig.add_subplot(111)
    for score, sub in summary.groupby("score", sort=True):
        sub = sub.sort_values(x_col)
        ax.errorbar(sub[x_col], sub["mean"], yerr=sub["se"], marker="o", capsize=3, label=str(score).upper())
    if target is not None and np.isfinite(target):
        ax.axhline(float(target), linestyle="--", linewidth=1, label=f"target={target:.2f}")
    ax.set_xlabel(r"$\delta$" if x_col == "delta" else "Number of clients K")
    if y_label == 'worst group cov':
        y_label = 'Worst-group coverage'
    elif y_label == 'normalized average set size':
        y_label = 'Normalized average set size'
    ax.set_ylabel(y_label)
    # ax.set_title(title)
    if log_x:
        ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def make_plots(results_csv: Path | str, out_dir: Path | str, coverage_metric: str = "worst_group_cov") -> List[Path]:
    out = ensure_dir(out_dir)
    df = pd.read_csv(results_csv)
    paths: List[Path] = []
    alpha = 0.1
    if "alpha" in df.columns and len(df):
        vals = pd.to_numeric(df["alpha"], errors="coerce").dropna()
        if len(vals):
            alpha = float(vals.iloc[0])

    cov_metric = str(coverage_metric)
    if cov_metric not in df.columns:
        raise ValueError(f"Unknown coverage metric '{cov_metric}'. Available columns include: {list(df.columns)}")

    for ablation, x_col, log_x in [("delta", "delta", True), ("k", "clients", False)]:
        cov = curve_summary(df, ablation, x_col, cov_metric)
        if not cov.empty:
            p = out / f"plot_{ablation}_{cov_metric}.png"
            _plot_curve(
                cov,
                x_col=x_col,
                y_label=cov_metric.replace("_", " "),
                title=f"GC-FCP {cov_metric.replace('_', ' ')} vs {x_col}",
                out_path=p,
                target=1.0 - alpha,
                log_x=log_x,
            )
            paths.append(p)

        ns = curve_summary(df, ablation, x_col, "norm_avg_set_size")
        if not ns.empty:
            p = out / f"plot_{ablation}_norm_set_size.png"
            _plot_curve(
                ns,
                x_col=x_col,
                y_label="normalized average set size",
                title=f"GC-FCP normalized set size vs {x_col}",
                out_path=p,
                target=None,
                log_x=log_x,
            )
            paths.append(p)

    return paths


# -------------------------
# CLI
# -------------------------


def add_common_run_args(p: argparse.ArgumentParser) -> None:
    # Defaults match the ImageNet rebuttal setup unless overridden.
    p.add_argument("--dataset", type=str, default="imagenet", choices=["cifar10", "cifar100", "imagenet"])
    p.add_argument("--imagenet_cache", type=str, default="data/imagenet_resnet50_val_probs.npz")
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--imagenet_val_dir", type=str, default="")
    p.add_argument("--imagenet_class_index_json", type=str, default="")
    p.add_argument("--imagenet_meta_map", type=str, default="")
    p.add_argument("--imagenet_model", type=str, default="resnet50", choices=["resnet50"])

    p.add_argument("--scores", type=str, default="thr", help="Comma-separated scores, e.g. thr,raps,aps.")
    p.add_argument("--times", type=int, default=100)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--n_cal", type=int, default=40000)
    p.add_argument("--n_test", type=int, default=10000)
    p.add_argument("--base_seed", type=int, default=114514)
    p.add_argument("--out_dir", type=str, default="results/rebuttal_ablation")
    p.add_argument("--quiet", action="store_true")

    p.add_argument("--clients", type=int, default=50, help="K used for group and delta ablations.")
    p.add_argument("--client_split", type=str, default="dirichlet", choices=["auto", "label", "dirichlet"])
    p.add_argument("--dirichlet_beta", type=float, default=0.3)
    p.add_argument("--min_client_count", type=int, default=200)

    p.add_argument("--group_mode", type=str, default="imagenet_semantic_ambiguity")
    p.add_argument("--x_feature", type=str, default="confidence", choices=["pred_label", "confidence", "entropy"])
    p.add_argument("--top_r", type=int, default=2)
    p.add_argument("--ambiguity_score", type=str, default="margin", choices=["margin", "entropy", "confidence"])
    p.add_argument("--ambiguity_bins", type=int, default=5)
    p.add_argument("--ambiguity_binning", type=str, default="quantile", choices=["quantile", "uniform"])

    p.add_argument("--tdigest_K", type=int, default=25)
    p.add_argument("--group_rule", type=str, default="min", choices=["min", "max", "random"])
    p.add_argument("--lp_max_iter", type=int, default=35)
    p.add_argument("--n_jobs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--raps_lambda", type=float, default=0.01)
    p.add_argument("--raps_k_reg", type=int, default=5)

    p.add_argument("--coverage_metric", type=str, default="worst_group_cov",
                   help="Coverage metric plotted for delta/K curves: worst_group_cov, avg_group_cov, or marg_cov.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GC-FCP no-exact ablation experiments.")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="Run selected ablations and save raw results, tables, and plots.")
    add_common_run_args(run)
    run.add_argument("--ablation", type=str, default="all", choices=["group", "delta", "k", "all"])
    run.add_argument("--group_designs", type=str, default="semantic,ambiguity,semantic_ambiguity",
                     help="Comma-separated group designs for ablation A.")
    run.add_argument("--group_delta", type=float, default=250.0, help="Delta used for group-design ablation.")
    run.add_argument("--delta_values", type=str, default="25,50,250,500,1000,2500",
                     help="Delta grid for compression ablation.")
    run.add_argument("--delta_group_design", type=str, default="semantic_ambiguity")
    run.add_argument("--k_values", type=str, default="1,10,20,30,40,50",
                     help="Client-count grid for federated aggregation ablation.")
    run.add_argument("--k_delta", type=float, default=250.0, help="Delta used for K ablation.")
    run.add_argument("--k_group_design", type=str, default="semantic_ambiguity")

    tables = sub.add_parser("tables", help="Generate ablation tables from saved ablation_raw_results.csv.")
    tables.add_argument("--results_csv", type=str, required=True)
    tables.add_argument("--out_dir", type=str, default="")

    plots = sub.add_parser("plots", help="Generate ablation plots from saved ablation_raw_results.csv.")
    plots.add_argument("--results_csv", type=str, required=True)
    plots.add_argument("--out_dir", type=str, default="")
    plots.add_argument("--coverage_metric", type=str, default="worst_group_cov")

    report = sub.add_parser("report", help="Generate both tables and plots from saved results.")
    report.add_argument("--results_csv", type=str, required=True)
    report.add_argument("--out_dir", type=str, default="")
    report.add_argument("--coverage_metric", type=str, default="worst_group_cov")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        args.command = "run"

    if args.command == "run":
        run_selected_ablations(args)
    elif args.command == "tables":
        results_csv = Path(args.results_csv)
        out_dir = Path(args.out_dir) if args.out_dir else results_csv.parent
        tables = make_tables(results_csv, out_dir)
        for name, tab in tables.items():
            print("\n" + "=" * 80)
            print(f"Ablation table: {name}")
            print("=" * 80)
            print(tab.to_markdown(index=False))
    elif args.command == "plots":
        results_csv = Path(args.results_csv)
        out_dir = Path(args.out_dir) if args.out_dir else results_csv.parent
        paths = make_plots(results_csv, out_dir, coverage_metric=args.coverage_metric)
        for p in paths:
            print(f"Saved {p}")
    elif args.command == "report":
        results_csv = Path(args.results_csv)
        out_dir = Path(args.out_dir) if args.out_dir else results_csv.parent
        make_tables(results_csv, out_dir)
        paths = make_plots(results_csv, out_dir, coverage_metric=args.coverage_metric)
        for p in paths:
            print(f"Saved {p}")
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate paper-style Table 2 / Table 4 from realdata_rebuttal_v4.py raw_results.csv.

Output columns mimic the original paper tables:
  Methods | Marginal coverage | G1 | ... | Comp. speedup
where each group cell is: coverage +/- se (set size +/- se).

This script expects raw_results.csv produced by realdata_rebuttal_v3_3.py or later,
which contains group_cov_json, group_size_json, and group_avg_set_size_json.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def parse_json_array(x: Any) -> np.ndarray:
    if isinstance(x, str):
        try:
            return np.asarray(json.loads(x), dtype=float)
        except Exception:
            return np.asarray([], dtype=float)
    if isinstance(x, (list, tuple, np.ndarray)):
        return np.asarray(x, dtype=float)
    return np.asarray([], dtype=float)


def mean_se(vals: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=1) / math.sqrt(arr.size))


def fmt_mean_se(mean: float, se: float, digits: int = 3) -> str:
    if not np.isfinite(mean):
        return "N/A"
    if not np.isfinite(se) or abs(se) < 1e-15:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} +/- {se:.{digits}f}"


def fmt_group(cov_mean: float, cov_se: float, size_mean: float, size_se: float) -> str:
    return f"{fmt_mean_se(cov_mean, cov_se, 3)} ({fmt_mean_se(size_mean, size_se, 2)})"


def method_order_for(table: str, df: pd.DataFrame) -> List[str]:
    # Original Table 2 order from the paper.
    if table == "table2":
        base = ["cp", "condcp", "naive_gcfcp", "centralized_gcfcp", "fedcp"]
    elif table == "table4":
        base = ["cp", "fedcp"]
    else:
        base = ["cp", "condcp", "naive_gcfcp", "centralized_gcfcp", "fedcp", "mondrian", "fedcf", "iw_fcp"]

    gcfcp_keys = [str(k) for k in df["method_key"].dropna().unique() if str(k).startswith("gcfcp_delta_")]

    def delta_value(k: str) -> float:
        s = k.replace("gcfcp_delta_", "")
        try:
            return float(s)
        except Exception:
            return float("inf")

    gcfcp_keys = sorted(gcfcp_keys, key=delta_value)
    keys = []
    available = set(str(k) for k in df["method_key"].dropna().unique())
    for k in base + gcfcp_keys:
        # centralized_gcfcp is stored as naive_gcfcp in realdata_rebuttal_v4.py.
        if k in available and k not in keys:
            keys.append(k)
    return keys


def display_method_name(method_key: str, method_name: str, table: str) -> str:
    if table == "table2" and method_key == "naive_gcfcp":
        return "Centralized GC-FCP"
    if method_key.startswith("gcfcp_delta_"):
        delta = method_key.replace("gcfcp_delta_", "")
        return f"GC-FCP (delta={delta})"
    return str(method_name)


def get_num_groups(sub: pd.DataFrame) -> int:
    max_g = 0
    for _, row in sub.iterrows():
        cov = parse_json_array(row.get("group_cov_json", "[]"))
        sz = parse_json_array(row.get("group_avg_set_size_json", "[]"))
        max_g = max(max_g, cov.size, sz.size)
    return max_g


def build_paper_table(
    raw_csv: Path,
    table: str = "auto",
    score: str = "thr",
    method_keys: Optional[Iterable[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(raw_csv)
    score = score.lower()
    work = df[df["score"].str.lower().eq(score)].copy()
    if work.empty:
        raise ValueError(f"No rows found for score={score!r} in {raw_csv}")

    if table == "auto":
        dataset = str(work["dataset"].iloc[0]).lower() if "dataset" in work.columns else ""
        table = "table2" if dataset == "cifar10" else ("table4" if dataset == "pathmnist" else "generic")

    if method_keys:
        keys = [k.strip() for k in method_keys if k.strip()]
    else:
        keys = method_order_for(table, work)

    rows = []
    numeric_rows = []
    n_groups = get_num_groups(work)

    for key in keys:
        sub = work[work["method_key"].astype(str).eq(key)]
        if sub.empty:
            continue
        method_name = display_method_name(key, sub["method"].iloc[0], table)
        row = {"Methods": method_name}
        nrow = {"method_key": key, "Methods": method_name}

        m, se = mean_se(pd.to_numeric(sub["marg_cov"], errors="coerce"))
        row["Marginal coverage"] = fmt_mean_se(m, se, 3)
        nrow["marg_cov_mean"] = m
        nrow["marg_cov_se"] = se

        for g in range(n_groups):
            cov_vals = []
            size_vals = []
            count_vals = []
            for _, r in sub.iterrows():
                cov = parse_json_array(r.get("group_cov_json", "[]"))
                group_size = parse_json_array(r.get("group_size_json", "[]"))
                group_set = parse_json_array(r.get("group_avg_set_size_json", "[]"))
                if g < cov.size and np.isfinite(cov[g]):
                    cov_vals.append(float(cov[g]))
                if g < group_set.size and np.isfinite(group_set[g]):
                    size_vals.append(float(group_set[g]))
                if g < group_size.size and np.isfinite(group_size[g]):
                    count_vals.append(float(group_size[g]))
            cov_m, cov_se = mean_se(cov_vals)
            sz_m, sz_se = mean_se(size_vals)
            cnt_m, cnt_se = mean_se(count_vals)
            col = f"G{g + 1}"
            row[col] = fmt_group(cov_m, cov_se, sz_m, sz_se)
            nrow[f"G{g + 1}_cov_mean"] = cov_m
            nrow[f"G{g + 1}_cov_se"] = cov_se
            nrow[f"G{g + 1}_set_size_mean"] = sz_m
            nrow[f"G{g + 1}_set_size_se"] = sz_se
            nrow[f"G{g + 1}_n_mean"] = cnt_m
            nrow[f"G{g + 1}_n_se"] = cnt_se

        if "comp_speedup" in sub.columns:
            sp_m, sp_se = mean_se(pd.to_numeric(sub["comp_speedup"], errors="coerce"))
            if np.isfinite(sp_m):
                row["Comp. speedup"] = fmt_mean_se(sp_m, sp_se, 2) + "x"
            else:
                row["Comp. speedup"] = "N/A"
            nrow["comp_speedup_mean"] = sp_m
            nrow["comp_speedup_se"] = sp_se
        rows.append(row)
        numeric_rows.append(nrow)

    out = pd.DataFrame(rows)
    numeric = pd.DataFrame(numeric_rows)
    preferred_cols = ["Methods", "Marginal coverage"] + [f"G{i+1}" for i in range(n_groups)] + ["Comp. speedup"]
    out = out[[c for c in preferred_cols if c in out.columns]]
    return out, numeric


def main() -> None:
    p = argparse.ArgumentParser(description="Generate paper-style Table 2/Table 4 with standard errors from raw_results.csv.")
    p.add_argument("--results_csv", required=True, help="Path to raw_results.csv produced by realdata_rebuttal_v4.py")
    p.add_argument("--table", default="auto", choices=["auto", "table2", "table4", "generic"])
    p.add_argument("--score", default="thr", help="Score to tabulate; paper tables use thr.")
    p.add_argument("--methods", default="", help="Optional comma-separated method_key order/selection.")
    p.add_argument("--out_prefix", default="", help="Output prefix. Defaults to <raw_results_dir>/<table>_<score>_stderr")
    args = p.parse_args()

    raw_csv = Path(args.results_csv)
    methods = [x.strip() for x in args.methods.split(",") if x.strip()] if args.methods else None
    table, numeric = build_paper_table(raw_csv, table=args.table, score=args.score, method_keys=methods)

    if args.out_prefix:
        prefix = Path(args.out_prefix)
    else:
        table_name = args.table if args.table != "auto" else "paper_table"
        prefix = raw_csv.parent / f"{table_name}_{args.score.lower()}_stderr"
    prefix.parent.mkdir(parents=True, exist_ok=True)

    table.to_csv(prefix.with_suffix(".csv"), index=False)
    numeric.to_csv(Path(str(prefix) + "_numeric.csv"), index=False)
    with open(prefix.with_suffix(".md"), "w", encoding="utf-8") as f:
        f.write(table.to_markdown(index=False))
        f.write("\n")

    print(table.to_markdown(index=False))
    print(f"\nSaved:\n  {prefix.with_suffix('.csv')}\n  {prefix.with_suffix('.md')}\n  {Path(str(prefix) + '_numeric.csv')}")


if __name__ == "__main__":
    main()

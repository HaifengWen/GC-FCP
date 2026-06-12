#!/usr/bin/env python3
"""
Rebuttal experiments for GC-FCP.

This script is self-contained and implements the additional experiment suite
requested for the UAI rebuttal:

  * datasets: synthetic regression, CIFAR-10, CIFAR-100, ImageNet-1K;
  * scores: THR/HPS, APS, RAPS for classification; absolute residual for synthetic;
  * baselines: CP, FedCP, Mondrian-FedCP, FedCF-style, importance-weighted FCP,
    Centralized CondCP (optional), Naive GC-FCP / Centralized GC-FCP (optional),
    and compressed GC-FCP;
  * efficiency: model/probability caching across scores, optional persistent
    ImageNet probability cache, and parallel LP solves over unique test
    group-membership patterns via --n_jobs;
  * tables: main THR+RAPS table, smaller APS table, and three HC6E ablation tables
    for group conditioning, compression, and federated aggregation;
  * raw outputs: per-MC group coverage, group counts, per-group average set size,
    and per-group normalized average set size.

Important implementation notes
------------------------------
1. The command-line --deltas are the paper/manuscript T-Digest compression
   parameters. The Python tdigest package uses an inverse-style parameter in the
   provided scripts. Therefore this script calls TDigest(delta=tdigest_K/delta).
2. If the tdigest package is unavailable, the script falls back to a deterministic
   weighted quantile coreset with approximately `delta` centroids per atom/group.
   This fallback is used only to keep the script runnable in minimal environments.
3. FedCF-style and importance-weighted FCP are adapted to this paper's group-
   conditional setting by calibrating original groups, not atoms, and by evaluating
   coverage on original groups. For overlapping groups, the default assignment rule
   is `min`: a sample belonging to multiple groups is assigned to the active group
   with the smallest group index. This matches the deterministic convention used
   for the group-level baselines in realdata_additional.py.
4. Naive GC-FCP and Centralized GC-FCP are identical at the algorithmic level:
   they solve the federated weighted augmented quantile regression exactly with
   all atom-stratified scores and no compression.
5. For ImageNet-1K, use the `cache_imagenet` subcommand once to precompute
   labels and pretrained-model probabilities. Subsequent `run` commands can
   pass --imagenet_cache to avoid repeated ImageNet image decoding and
   ResNet-50 inference.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
import warnings
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.stats import truncnorm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Keep numerical libraries from oversubscribing by default.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")


# -------------------------
# Generic helper utilities
# -------------------------

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # noqa: F401
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def parse_csv_list(s: str, typ=float) -> List[Any]:
    out: List[Any] = []
    for part in (s or "").split(","):
        part = part.strip()
        if part:
            out.append(typ(part))
    return out


def normalize_pi(pi: Optional[Sequence[float]], K: int) -> np.ndarray:
    if pi is None or len(pi) == 0:
        return np.ones(K, dtype=float) / K
    arr = np.array(pi, dtype=float)
    if len(arr) != K:
        raise ValueError(f"Expected {K} mixture weights but got {len(arr)}.")
    if np.any(arr < 0):
        raise ValueError("Mixture weights must be non-negative.")
    s = float(arr.sum())
    if s <= 0:
        raise ValueError("Mixture weights must sum to a positive value.")
    return arr / s


def fmt_delta(delta: float) -> str:
    return str(int(delta)) if float(delta).is_integer() else f"{delta:g}"


def weighted_quantile(values: np.ndarray, weights: Optional[np.ndarray], q: float) -> float:
    """Smallest value whose cumulative normalized weight is at least q."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("inf")
    if weights is None:
        weights = np.ones(values.size, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[mask]
    weights = weights[mask]
    if values.size == 0 or weights.sum() <= 0:
        return float("inf")
    order = np.argsort(values)
    vs = values[order]
    ws = weights[order]
    target = float(q) * float(ws.sum())
    idx = int(np.searchsorted(np.cumsum(ws), target, side="left"))
    idx = min(max(idx, 0), len(vs) - 1)
    return float(vs[idx])


def weighted_quantile_target(values: np.ndarray, weights: np.ndarray, target_mass: float) -> float:
    """Smallest value whose cumulative unnormalized weight is at least target_mass."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.size == 0:
        return float("inf")
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[mask]
    weights = weights[mask]
    if values.size == 0:
        return float("inf")
    order = np.argsort(values)
    vs = values[order]
    ws = weights[order]
    cum = np.cumsum(ws)
    if float(cum[-1]) < float(target_mass):
        return float("inf")
    idx = int(np.searchsorted(cum, target_mass, side="left"))
    idx = min(max(idx, 0), len(vs) - 1)
    return float(vs[idx])


def split_cp_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample split CP quantile with ceil((n+1)(1-alpha))/n correction."""
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    n = len(scores)
    if n == 0:
        return float("inf")
    rank = int(math.ceil((n + 1) * (1.0 - alpha)))
    if rank > n:
        return float("inf")
    return float(np.partition(scores, rank - 1)[rank - 1])


def row_tuples(mat: np.ndarray) -> List[Tuple[int, ...]]:
    return [tuple(int(v) for v in row) for row in np.asarray(mat)]


def unique_rows(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return unique rows and inverse indices."""
    mat = np.asarray(mat, dtype=int)
    if mat.ndim != 2:
        raise ValueError("Expected a 2-D array.")
    uniq, inv = np.unique(mat, axis=0, return_inverse=True)
    return uniq.astype(int), inv.astype(int)


def combine_group_thresholds(
    phi_test: np.ndarray,
    group_taus: np.ndarray,
    fallback_tau: float,
    rule: str = "min",
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Convert group-level thresholds into one threshold per test point.

    This function is used only by group-level baselines such as Mondrian-FedCP,
    FedCF-style, and IW-FCP. These baselines do not solve the atom-level CondCP
    problem, so a test sample in overlapping groups must be assigned to one group
    before selecting the group threshold.

    The default is rule="min", which chooses the active group with the smallest
    group index. This is an index rule, not a threshold-value rule:
      * rule="min" chooses min({g: x in G_g});
      * rule="max" chooses max({g: x in G_g});
      * rule="random" samples one active group uniformly.
    """
    phi_test = np.asarray(phi_test, dtype=int)
    group_taus = np.asarray(group_taus, dtype=float)
    rule = (rule or "min").lower()
    if rng is None:
        rng = np.random.default_rng(0)

    taus = np.full(phi_test.shape[0], float(fallback_tau), dtype=float)
    for i, row in enumerate(phi_test):
        active = np.where(row > 0)[0]
        if len(active) == 0:
            taus[i] = fallback_tau
        elif rule == "min":
            taus[i] = float(group_taus[int(active[0])])
        elif rule == "max":
            taus[i] = float(group_taus[int(active[-1])])
        elif rule == "random":
            taus[i] = float(group_taus[int(rng.choice(active))])
        else:
            raise ValueError(f"Unknown group assignment rule: {rule}")
    return taus

def safe_float(x: Any) -> float:
    try:
        y = float(x)
        if math.isnan(y):
            return float("nan")
        return y
    except Exception:
        return float("nan")


# -------------------------
# Parallel LP worker state
# -------------------------

_LP_SHARED: Dict[str, Any] = {}


def _lp_worker_init(
    phi_all: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
    w_test: float,
    alpha: float,
    score_upper: float,
    max_iter: int,
) -> None:
    """Store LP data once per worker process.

    Passing calibration arrays once via the process initializer is much cheaper
    than serializing them for every unique test membership pattern.
    """
    global _LP_SHARED
    _LP_SHARED = {
        "phi_all": np.asarray(phi_all, dtype=float),
        "scores": np.asarray(scores, dtype=float),
        "weights": np.asarray(weights, dtype=float),
        "w_test": float(w_test),
        "alpha": float(alpha),
        "score_upper": float(score_upper),
        "max_iter": int(max_iter),
    }


def _lp_worker_solve(phi_test: np.ndarray) -> float:
    return solve_lp_threshold_for_phi(
        np.asarray(phi_test, dtype=float),
        _LP_SHARED["phi_all"],
        _LP_SHARED["scores"],
        _LP_SHARED["weights"],
        _LP_SHARED["w_test"],
        _LP_SHARED["alpha"],
        _LP_SHARED["score_upper"],
        max_iter=_LP_SHARED["max_iter"],
    )


# -------------------------
# Weighted coreset / T-Digest
# -------------------------

def _try_tdigest_class():
    try:
        from tdigest import TDigest  # type: ignore
        return TDigest
    except Exception:
        return None


def compress_weighted_scores(
    scores: np.ndarray,
    weights: np.ndarray,
    paper_delta: float,
    tdigest_K: int = 25,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Compress weighted scores into centroids.

    The input `paper_delta` is the paper parameter. When the tdigest package is
    installed, the package argument is tdigest_K / paper_delta, matching the
    convention in the uploaded scripts.
    """
    scores = np.asarray(scores, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(scores) & np.isfinite(weights) & (weights > 0)
    scores = scores[mask]
    weights = weights[mask]
    if len(scores) == 0:
        return np.array([], dtype=float), np.array([], dtype=float), "empty"

    TDigest = _try_tdigest_class()
    if TDigest is not None:
        package_delta = float(tdigest_K) / float(paper_delta)
        td = TDigest(delta=package_delta, K=int(tdigest_K))
        for s, w in zip(scores, weights):
            td.update(float(s), float(w))
        cents = td.centroids_to_list()
        if len(cents) == 0:
            return np.array([], dtype=float), np.array([], dtype=float), "tdigest"
        means = np.array([c["m"] for c in cents], dtype=float)
        cweights = np.array([c["c"] for c in cents], dtype=float)
        return means, cweights, "tdigest"

    # Fallback: deterministic equal-mass weighted quantile coreset.
    order = np.argsort(scores)
    s_sorted = scores[order]
    w_sorted = weights[order]
    total_w = float(w_sorted.sum())
    m = max(1, min(len(s_sorted), int(math.ceil(float(paper_delta)))))
    target = total_w / m
    means: List[float] = []
    cweights: List[float] = []
    acc_w = 0.0
    acc_sw = 0.0
    for s, w in zip(s_sorted, w_sorted):
        if acc_w > 0 and acc_w + w > target and len(means) < m - 1:
            means.append(acc_sw / acc_w)
            cweights.append(acc_w)
            acc_w = 0.0
            acc_sw = 0.0
        acc_w += float(w)
        acc_sw += float(s) * float(w)
    if acc_w > 0:
        means.append(acc_sw / acc_w)
        cweights.append(acc_w)
    return np.array(means, dtype=float), np.array(cweights, dtype=float), "fallback_quantile_coreset"


def compress_by_atoms_federated(
    scores_by_client: List[np.ndarray],
    weights_by_client: List[np.ndarray],
    phi_by_client: List[np.ndarray],
    paper_delta: float,
    tdigest_K: int = 25,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, str]:
    """Construct GC-FCP coreset by local atom compression and server merging."""
    all_keys = sorted(set(k for phi in phi_by_client for k in row_tuples(phi)))
    d = phi_by_client[0].shape[1] if len(phi_by_client) > 0 and len(phi_by_client[0].shape) == 2 else 0

    local_centroids_by_atom: Dict[Tuple[int, ...], List[Tuple[np.ndarray, np.ndarray]]] = {k: [] for k in all_keys}
    comm = 0
    impl = "empty"

    for s_k, w_k, phi_k in zip(scores_by_client, weights_by_client, phi_by_client):
        keys_k = row_tuples(phi_k)
        for key in all_keys:
            idx = np.array([kk == key for kk in keys_k], dtype=bool)
            if not np.any(idx):
                continue
            means, cweights, impl_k = compress_weighted_scores(s_k[idx], w_k[idx], paper_delta, tdigest_K)
            impl = impl_k if impl == "empty" else impl
            if len(means) > 0:
                local_centroids_by_atom[key].append((means, cweights))
                comm += len(means)

    pseudo_s: List[float] = []
    pseudo_w: List[float] = []
    pseudo_phi: List[List[int]] = []

    for key, local_parts in local_centroids_by_atom.items():
        if not local_parts:
            continue
        means_all = np.concatenate([p[0] for p in local_parts])
        weights_all = np.concatenate([p[1] for p in local_parts])
        merged_s, merged_w, impl_m = compress_weighted_scores(means_all, weights_all, paper_delta, tdigest_K)
        if impl == "empty":
            impl = impl_m
        for s, w in zip(merged_s, merged_w):
            pseudo_s.append(float(s))
            pseudo_w.append(float(w))
            pseudo_phi.append(list(key))

    if len(pseudo_s) == 0:
        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.zeros((0, d), dtype=int),
            len(all_keys),
            0,
            impl,
        )

    return (
        np.array(pseudo_s, dtype=float),
        np.array(pseudo_w, dtype=float),
        np.array(pseudo_phi, dtype=int),
        len(all_keys),
        comm,
        impl,
    )


def group_digest_thresholds_federated(
    scores_by_client: List[np.ndarray],
    weights_by_client: List[np.ndarray],
    phi_by_client: List[np.ndarray],
    alpha: float,
    paper_delta: float,
    tdigest_K: int = 25,
) -> Tuple[np.ndarray, int, int, str]:
    """FedCF-style group thresholds with one federated digest per original group."""
    d = phi_by_client[0].shape[1]
    taus = np.full(d, float("inf"), dtype=float)
    comm = 0
    coreset = 0
    impl = "empty"

    for g in range(d):
        local_parts: List[Tuple[np.ndarray, np.ndarray]] = []
        for s_k, w_k, phi_k in zip(scores_by_client, weights_by_client, phi_by_client):
            idx = phi_k[:, g] > 0
            if not np.any(idx):
                continue
            means, cweights, impl_k = compress_weighted_scores(s_k[idx], w_k[idx], paper_delta, tdigest_K)
            impl = impl_k if impl == "empty" else impl
            if len(means) > 0:
                local_parts.append((means, cweights))
                comm += len(means)
        if not local_parts:
            continue
        means_all = np.concatenate([p[0] for p in local_parts])
        weights_all = np.concatenate([p[1] for p in local_parts])
        merged_s, merged_w, impl_m = compress_weighted_scores(means_all, weights_all, paper_delta, tdigest_K)
        if impl == "empty":
            impl = impl_m
        coreset += len(merged_s)
        taus[g] = weighted_quantile(merged_s, merged_w, 1.0 - alpha)

    return taus, comm, coreset, impl


# -------------------------
# Experiment data containers
# -------------------------

@dataclass
class ExperimentData:
    dataset: str
    score: str
    alpha: float
    group_names: List[str]
    client_scores: List[np.ndarray]
    client_phi: List[np.ndarray]
    client_labels: Optional[List[np.ndarray]]
    client_x_scalar: List[np.ndarray]
    pi: np.ndarray
    test_scores_true: np.ndarray
    test_scores_all: Optional[np.ndarray]
    test_phi: np.ndarray
    test_labels: Optional[np.ndarray]
    set_kind: str  # "classification" or "regression"
    score_upper: float
    metadata: Dict[str, Any]

    @property
    def K(self) -> int:
        return len(self.client_scores)

    @property
    def n_cal(self) -> int:
        return int(sum(len(s) for s in self.client_scores))

    @property
    def n_test(self) -> int:
        return int(len(self.test_scores_true))

    @property
    def n_groups(self) -> int:
        return int(len(self.group_names))


@dataclass
class MethodRecord:
    seed: int
    mc: int
    dataset: str
    score: str
    method_key: str
    method: str
    alpha: float
    marg_cov: float
    worst_group_cov: float
    avg_group_cov: float
    max_group_gap: float
    avg_set_size: float
    norm_avg_set_size: float
    atoms: int
    coreset: int
    comm: int
    comp_speedup: float
    n_cal: int
    n_test: int
    n_groups: int
    runtime_sec: float
    group_cov_json: str
    group_size_json: str
    group_avg_set_size_json: str
    group_norm_avg_set_size_json: str
    details_json: str


# -------------------------
# Group construction
# -------------------------

CIFAR100_FINE_TO_COARSE = np.array([
    4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
    3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
    6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
    0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
    5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
    16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
    10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
    2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
    16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
    18, 1, 2, 15, 6, 0, 17, 8, 14, 13
], dtype=int)

CIFAR100_COARSE_NAMES = [
    "aquatic_mammals", "fish", "flowers", "food_containers", "fruit_vegetables",
    "household_electrical", "household_furniture", "insects", "large_carnivores",
    "large_manmade_outdoor", "large_natural_outdoor", "large_omnivores_herbivores",
    "medium_mammals", "noninsect_invertebrates", "people", "reptiles",
    "small_mammals", "trees", "vehicles_1", "vehicles_2"
]


def phi_from_intervals(x: np.ndarray, groups: List[Tuple[float, float]]) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    phi = np.zeros((len(x), len(groups)), dtype=int)
    for g, (lo, hi) in enumerate(groups):
        # Include the right end for the last interval.
        if g == len(groups) - 1:
            phi[:, g] = ((x >= lo) & (x <= hi)).astype(int)
        else:
            phi[:, g] = ((x >= lo) & (x < hi)).astype(int)
    return phi


def phi_cifar100_topr(probs: np.ndarray, top_r: int = 2) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    coarse = np.zeros((probs.shape[0], 20), dtype=float)
    for fine in range(100):
        coarse[:, CIFAR100_FINE_TO_COARSE[fine]] += probs[:, fine]
    top = np.argsort(-coarse, axis=1)[:, :top_r]
    phi = np.zeros((probs.shape[0], 20), dtype=int)
    for i in range(probs.shape[0]):
        phi[i, top[i]] = 1
    return phi


IMAGENET_META_NAMES = [
    "dogs_wolves_foxes",
    "other_mammals",
    "birds",
    "reptiles_amphibians_fish",
    "insects_arthropods_invertebrates",
    "vehicles",
    "instruments_tools_devices",
    "furniture_household_objects",
    "food_plants_fungi",
    "scenes_structures_misc_artifacts",
]


def _keyword_match(label: str, keywords: Sequence[str]) -> bool:
    ll = " " + label.lower().replace("_", " ").replace("-", " ") + " "
    return any((" " + kw.lower().replace("_", " ").replace("-", " ") + " ") in ll for kw in keywords)


def get_imagenet_categories(model_name: str = "resnet50") -> List[str]:
    import torchvision.models as models
    if str(model_name).lower() != "resnet50":
        raise ValueError("Only --imagenet_model resnet50 is currently supported.")
    weights = models.ResNet50_Weights.DEFAULT
    return list(weights.meta.get("categories", []))


def load_imagenet_meta_map_file(path: str, categories: Sequence[str]) -> Optional[np.ndarray]:
    """Load optional ImageNet fine-to-meta mapping.

    Supported JSON formats:
      * list of 1000 integer group ids in [0,9];
      * dict {"0": 3, "1": 4, ...} mapping class index to group id;
      * dict {"n01440764": 3, "tench": 3, ...} mapping folder/category names to group id;
      * dict {"group_name": [indices or category names or wnids, ...], ...}.
    Group ids may also be meta-group names from IMAGENET_META_NAMES.
    """
    if not path:
        return None
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"ImageNet meta map not found: {path}")
    with open(path_obj, "r", encoding="utf-8") as f:
        obj = json.load(f)

    def to_gid(v: Any) -> int:
        if isinstance(v, str):
            vv = v.strip()
            if vv in IMAGENET_META_NAMES:
                return IMAGENET_META_NAMES.index(vv)
            return int(vv)
        return int(v)

    mapping = np.full(1000, 9, dtype=int)
    cat_to_idx = {str(c).lower(): i for i, c in enumerate(categories)}

    if isinstance(obj, list):
        if len(obj) != 1000:
            raise ValueError("ImageNet meta-map list must have length 1000.")
        mapping = np.array([to_gid(v) for v in obj], dtype=int)
    elif isinstance(obj, dict):
        if any(k in IMAGENET_META_NAMES for k in obj.keys()) or all(isinstance(v, list) for v in obj.values()):
            for group_key, members in obj.items():
                if not isinstance(members, list):
                    continue
                gid = to_gid(group_key)
                for m in members:
                    if isinstance(m, int) or (isinstance(m, str) and m.strip().isdigit()):
                        idx = int(m)
                        if 0 <= idx < 1000:
                            mapping[idx] = gid
                    else:
                        idx = cat_to_idx.get(str(m).lower())
                        if idx is not None:
                            mapping[idx] = gid
        else:
            for k, v in obj.items():
                gid = to_gid(v)
                if str(k).isdigit():
                    idx = int(k)
                else:
                    idx = cat_to_idx.get(str(k).lower(), None)
                if idx is not None and 0 <= idx < 1000:
                    mapping[idx] = gid
    else:
        raise ValueError("Unsupported ImageNet meta-map JSON format.")

    if np.any((mapping < 0) | (mapping >= 10)):
        raise ValueError("ImageNet meta group ids must be in [0,9].")
    return mapping.astype(int)


def build_imagenet_meta_mapping(categories: Sequence[str], map_file: str = "") -> Tuple[np.ndarray, List[str], str]:
    """Build a 1000 -> 10 semantic meta-group mapping.

    The default mapping is a lightweight keyword heuristic over TorchVision's
    ImageNet-1K category names. For manuscript-quality experiments, users may
    provide --imagenet_meta_map to use a curated mapping.
    """
    loaded = load_imagenet_meta_map_file(map_file, categories)
    if loaded is not None:
        return loaded, IMAGENET_META_NAMES[:], "user_json"

    dog_kw = ["dog", "hound", "terrier", "retriever", "spaniel", "shepherd", "setter", "poodle", "collie", "mastiff", "schnauzer", "chihuahua", "pekinese", "shih", "papillon", "ridgeback", "beagle", "bloodhound", "borzoi", "wolf", "fox", "coyote", "dingo", "malamute", "husky", "kelpie", "corgi", "dachshund", "pinscher", "rottweiler", "dalmatian", "pomeranian", "chow", "basenji", "pug", "leonberg", "newfoundland", "boxer", "great dane", "saint bernard", "schipperke"]
    mammal_kw = ["cat", "tiger", "lion", "leopard", "jaguar", "cheetah", "bear", "elephant", "monkey", "ape", "lemur", "panda", "koala", "sloth", "rabbit", "hare", "squirrel", "marmot", "beaver", "porcupine", "hamster", "mouse", "marten", "otter", "badger", "skunk", "weasel", "meerkat", "mongoose", "armadillo", "wombat", "kangaroo", "horse", "zebra", "hog", "pig", "boar", "ox", "cow", "buffalo", "bison", "sheep", "ram", "goat", "llama", "camel", "deer", "antelope", "impala", "gazelle", "hippopotamus", "rhinoceros", "whale", "seal", "sea lion", "dugong"]
    bird_kw = ["bird", "cock", "hen", "ostrich", "finch", "junco", "bunting", "robin", "bulbul", "jay", "magpie", "chickadee", "ouzel", "kite", "eagle", "vulture", "owl", "grouse", "quail", "partridge", "macaw", "parrot", "lorikeet", "coucal", "bee eater", "hornbill", "hummingbird", "jacamar", "toucan", "drake", "merganser", "goose", "swan", "flamingo", "stork", "heron", "bittern", "crane", "limpkin", "gallinule", "coot", "bustard", "turnstone", "redshank", "dowitcher", "oystercatcher", "pelican", "penguin", "albatross"]
    reptile_fish_kw = ["fish", "shark", "ray", "eel", "coho", "tench", "goldfish", "barracouta", "sturgeon", "gar", "puffer", "turtle", "tortoise", "terrapin", "lizard", "iguana", "chameleon", "gecko", "dragon", "crocodile", "alligator", "snake", "viper", "cobra", "python", "boa", "frog", "toad", "salamander", "newt", "axolotl"]
    invert_kw = ["insect", "bee", "ant", "fly", "grasshopper", "cricket", "walking stick", "cockroach", "mantis", "cicada", "leafhopper", "lacewing", "dragonfly", "damselfly", "butterfly", "ladybug", "beetle", "weevil", "spider", "tick", "centipede", "scorpion", "trilobite", "crayfish", "lobster", "crab", "isopod", "barnacle", "snail", "slug", "worm", "anemone", "coral", "jellyfish", "sea urchin", "sea cucumber", "starfish"]
    vehicle_kw = ["car", "cab", "taxi", "limousine", "jeep", "ambulance", "wagon", "convertible", "racer", "minivan", "pickup", "truck", "engine", "snowplow", "bus", "trolleybus", "streetcar", "train", "locomotive", "motorcycle", "moped", "bicycle", "tricycle", "boat", "ship", "schooner", "canoe", "kayak", "speedboat", "gondola", "submarine", "aircraft", "airplane", "airliner", "airship", "spacecraft", "scooter", "tractor"]
    instrument_tool_kw = ["instrument", "guitar", "banjo", "cello", "violin", "piano", "organ", "harmonica", "drum", "sax", "flute", "oboe", "accordion", "computer", "laptop", "monitor", "keyboard", "printer", "modem", "router", "phone", "cellular", "ipod", "camera", "lens", "microscope", "oscilloscope", "radio", "television", "cassette", "tape player", "cd player", "joystick", "remote control", "drill", "hammer", "wrench", "screwdriver", "saw", "plane", "axe", "shovel", "plow", "machine", "projector"]
    furniture_kw = ["chair", "sofa", "couch", "bed", "table", "desk", "cradle", "crib", "wardrobe", "cabinet", "chest", "bookcase", "lamp", "lampshade", "pillow", "quilt", "curtain", "shade", "toilet", "bath", "bathtub", "sink", "washbasin", "refrigerator", "stove", "microwave", "dishwasher", "washer"]
    food_plant_kw = ["fruit", "vegetable", "apple", "orange", "lemon", "fig", "pineapple", "banana", "mushroom", "corn", "broccoli", "cauliflower", "zucchini", "cucumber", "artichoke", "pepper", "acorn", "rose", "daisy", "orchid", "tulip", "sunflower", "cabbage", "strawberry", "pomegranate", "hay", "carbonara", "hotdog", "pizza", "burrito", "espresso", "chocolate", "meat loaf", "cheeseburger", "ice cream", "pretzel", "bagel", "guacamole", "consomme", "trifle", "wine bottle", "eggnog", "bakery"]

    mapping = np.full(1000, 9, dtype=int)
    for i, cat in enumerate(categories):
        lab = str(cat).lower().replace("_", " ").replace("-", " ")
        if _keyword_match(lab, dog_kw):
            mapping[i] = 0
        elif _keyword_match(lab, mammal_kw):
            mapping[i] = 1
        elif _keyword_match(lab, bird_kw):
            mapping[i] = 2
        elif _keyword_match(lab, reptile_fish_kw):
            mapping[i] = 3
        elif _keyword_match(lab, invert_kw):
            mapping[i] = 4
        elif _keyword_match(lab, vehicle_kw):
            mapping[i] = 5
        elif _keyword_match(lab, instrument_tool_kw):
            mapping[i] = 6
        elif _keyword_match(lab, furniture_kw):
            mapping[i] = 7
        elif _keyword_match(lab, food_plant_kw):
            mapping[i] = 8
        else:
            mapping[i] = 9
    return mapping.astype(int), IMAGENET_META_NAMES[:], "keyword_heuristic"


def phi_fine_to_meta_topr(probs: np.ndarray, fine_to_meta: np.ndarray, n_meta: int, top_r: int = 2) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    fine_to_meta = np.asarray(fine_to_meta, dtype=int)
    meta = np.zeros((probs.shape[0], int(n_meta)), dtype=float)
    for fine in range(probs.shape[1]):
        g = int(fine_to_meta[fine]) if fine < len(fine_to_meta) else int(n_meta) - 1
        if 0 <= g < int(n_meta):
            meta[:, g] += probs[:, fine]
    top = np.argsort(-meta, axis=1)[:, : int(top_r)]
    phi = np.zeros((probs.shape[0], int(n_meta)), dtype=int)
    for i in range(probs.shape[0]):
        phi[i, top[i]] = 1
    return phi


def phi_imagenet_meta_topr(probs: np.ndarray, fine_to_meta: np.ndarray, top_r: int = 2) -> np.ndarray:
    return phi_fine_to_meta_topr(probs, fine_to_meta, n_meta=10, top_r=top_r)


def imagenet_meta_probabilities(probs: np.ndarray, fine_to_meta: np.ndarray, n_meta: int = 10) -> np.ndarray:
    """Aggregate ImageNet fine-class probabilities into semantic meta-group probabilities."""
    probs = np.asarray(probs, dtype=float)
    fine_to_meta = np.asarray(fine_to_meta, dtype=int)
    meta = np.zeros((probs.shape[0], int(n_meta)), dtype=float)
    for fine in range(probs.shape[1]):
        g = int(fine_to_meta[fine]) if fine < len(fine_to_meta) else int(n_meta) - 1
        if 0 <= g < int(n_meta):
            meta[:, g] += probs[:, fine]
    return meta


def ambiguity_score_from_probs(probs: np.ndarray, mode: str = "margin") -> np.ndarray:
    """Return a scalar ambiguity / difficulty score from class probabilities.

    Supported modes:
      * margin: top-1 minus top-2 probability. Smaller values are more ambiguous.
      * entropy: predictive entropy. Larger values are more ambiguous.
      * confidence: top-1 probability. Smaller values are more ambiguous.
    """
    probs = np.asarray(probs, dtype=float)
    mode = (mode or "margin").lower()
    if probs.ndim != 2 or probs.shape[1] < 2:
        raise ValueError("Expected a 2D probability array with at least two classes.")
    top2 = np.partition(probs, kth=probs.shape[1] - 2, axis=1)[:, -2:]
    p1 = np.max(top2, axis=1)
    p2 = np.min(top2, axis=1)
    if mode == "margin":
        return (p1 - p2).astype(float)
    if mode == "entropy":
        return (-np.sum(probs * np.log(probs + 1e-12), axis=1)).astype(float)
    if mode == "confidence":
        return p1.astype(float)
    raise ValueError("ambiguity_score must be one of {'margin','entropy','confidence'}")


def make_scalar_bins(
    x_reference: np.ndarray,
    num_bins: int,
    binning: str = "quantile",
    value_range: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """Build monotone bin edges for scalar difficulty/ambiguity groups."""
    x_reference = np.asarray(x_reference, dtype=float)
    num_bins = int(num_bins)
    if num_bins <= 0:
        raise ValueError("num_bins must be positive.")
    binning = (binning or "quantile").lower()
    if x_reference.size == 0:
        raise ValueError("Cannot build bins from an empty reference array.")
    if binning == "quantile":
        qs = np.linspace(0.0, 1.0, num_bins + 1)
        edges = np.quantile(x_reference, qs).astype(float)
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = np.nextafter(edges[i - 1], np.inf)
        edges[0] = float(np.min(x_reference))
        edges[-1] = float(np.max(x_reference))
        if edges[-1] <= edges[0]:
            edges = np.linspace(edges[0], edges[0] + 1e-8, num_bins + 1)
        else:
            edges[-1] = np.nextafter(edges[-1], np.inf)
        return edges
    if binning == "uniform":
        if value_range is None:
            lo, hi = float(np.min(x_reference)), float(np.max(x_reference))
        else:
            lo, hi = float(value_range[0]), float(value_range[1])
        if hi <= lo:
            hi = lo + 1e-8
        edges = np.linspace(lo, hi, num_bins + 1)
        edges[-1] = np.nextafter(edges[-1], np.inf)
        return edges.astype(float)
    raise ValueError("ambiguity_binning must be one of {'quantile','uniform'}")


def phi_from_bin_edges(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """One-hot bin membership for scalar values and fixed edges."""
    x = np.asarray(x, dtype=float)
    edges = np.asarray(edges, dtype=float)
    if edges.ndim != 1 or len(edges) < 2:
        raise ValueError("edges must be a 1D array with length >= 2.")
    B = len(edges) - 1
    idx = np.searchsorted(edges[1:-1], x, side="right")
    idx = np.clip(idx, 0, B - 1)
    phi = np.zeros((len(x), B), dtype=int)
    phi[np.arange(len(x)), idx] = 1
    return phi


def phi_imagenet_semantic_ambiguity(
    probs_cal: np.ndarray,
    probs_test: np.ndarray,
    fine_to_meta: np.ndarray,
    meta_names: Sequence[str],
    num_bins: int = 5,
    ambiguity_score: str = "margin",
    ambiguity_binning: str = "quantile",
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
    """Build ImageNet semantic x ambiguity groups.

    The group family is the union of 10 top-1 semantic meta-groups and
    `num_bins` ambiguity/difficulty bins based on top-2 margin, entropy, or
    confidence. Each sample belongs to exactly one semantic group and exactly
    one ambiguity group, so the atoms correspond to semantic-difficulty
    intersections and are at most 10 * num_bins.
    """
    n_meta = int(len(meta_names))
    meta_cal = imagenet_meta_probabilities(probs_cal, fine_to_meta, n_meta=n_meta)
    meta_test = imagenet_meta_probabilities(probs_test, fine_to_meta, n_meta=n_meta)

    top_meta_cal = np.argmax(meta_cal, axis=1)
    top_meta_test = np.argmax(meta_test, axis=1)
    phi_sem_cal = np.zeros((probs_cal.shape[0], n_meta), dtype=int)
    phi_sem_test = np.zeros((probs_test.shape[0], n_meta), dtype=int)
    phi_sem_cal[np.arange(probs_cal.shape[0]), top_meta_cal] = 1
    phi_sem_test[np.arange(probs_test.shape[0]), top_meta_test] = 1

    amb_cal = ambiguity_score_from_probs(probs_cal, mode=ambiguity_score)
    amb_test = ambiguity_score_from_probs(probs_test, mode=ambiguity_score)
    ambiguity_score_l = (ambiguity_score or "margin").lower()
    if ambiguity_score_l in {"margin", "confidence"}:
        value_range = (0.0, 1.0)
    else:
        value_range = (0.0, float(np.log(probs_cal.shape[1]) + 1e-6))
    edges = make_scalar_bins(amb_cal, num_bins=num_bins, binning=ambiguity_binning, value_range=value_range)
    phi_amb_cal = phi_from_bin_edges(amb_cal, edges)
    phi_amb_test = phi_from_bin_edges(amb_test, edges)

    phi_cal = np.hstack([phi_sem_cal, phi_amb_cal]).astype(int)
    phi_test = np.hstack([phi_sem_test, phi_amb_test]).astype(int)
    group_names = [f"sem_{name}" for name in meta_names] + [f"{ambiguity_score_l}_bin_{i+1}" for i in range(int(num_bins))]
    meta: Dict[str, Any] = {
        "group_mode": "imagenet_semantic_ambiguity",
        "semantic_group_names": list(meta_names),
        "ambiguity_score": ambiguity_score_l,
        "ambiguity_binning": (ambiguity_binning or "quantile").lower(),
        "ambiguity_bins": int(num_bins),
        "ambiguity_edges": [float(v) for v in edges.tolist()],
        "active_atom_upper_bound": int(n_meta * int(num_bins)),
        "group_construction": "10 top-1 ImageNet semantic meta-groups plus ambiguity bins; each sample has one semantic and one ambiguity membership.",
    }
    return phi_cal, phi_test, group_names, meta


class RemappedTargetDataset:
    def __init__(self, base: Any, target_remap: Dict[int, int]):
        self.base = base
        self.target_remap = dict(target_remap)
        self.classes = getattr(base, "classes", None)
        self.class_to_idx = getattr(base, "class_to_idx", None)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[idx]
        return x, int(self.target_remap.get(int(y), int(y)))


def load_imagenet_class_index_json(path: str) -> Dict[str, int]:
    if not path:
        return {}
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"ImageNet class-index JSON not found: {path}")
    with open(path_obj, "r", encoding="utf-8") as f:
        obj = json.load(f)
    out: Dict[str, int] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).isdigit() and isinstance(v, (list, tuple)) and len(v) >= 1:
                idx = int(k)
                out[str(v[0])] = idx
                if len(v) >= 2:
                    out[str(v[1]).lower()] = idx
            else:
                try:
                    out[str(k)] = int(v)
                    out[str(k).lower()] = int(v)
                except Exception:
                    pass
    return out


def load_imagenet_dataset(data_root: str, val_dir: str, transform: Any, class_index_json: str = ""):
    """Load ImageNet-1K validation data.

    Preferred: torchvision.datasets.ImageNet(root=data_root, split="val") with
    standard meta files. Fallback: ImageFolder over --imagenet_val_dir or
    data_root/val. For ImageFolder with synset folders, provide
    --imagenet_class_index_json so targets match the pretrained model index.
    """
    import torchvision

    root = Path(data_root)
    candidate_val = Path(val_dir) if val_dir else root / "val"

    # If the user did not explicitly provide ImageFolder path, first try the
    # official torchvision ImageNet loader, which has the correct class-index
    # metadata when the devkit has been prepared.
    if not val_dir:
        try:
            return torchvision.datasets.ImageNet(root=str(root), split="val", transform=transform)
        except Exception:
            pass

    if val_dir or candidate_val.exists():
        ds_folder = torchvision.datasets.ImageFolder(root=str(candidate_val), transform=transform)
        class_names = list(ds_folder.classes)
        target_remap: Dict[int, int] = {}
        class_index = load_imagenet_class_index_json(class_index_json)

        if all(c.isdigit() and 0 <= int(c) < 1000 for c in class_names):
            for c, old_idx in ds_folder.class_to_idx.items():
                target_remap[int(old_idx)] = int(c)
        elif class_index:
            for c, old_idx in ds_folder.class_to_idx.items():
                key_options = [c, c.lower(), c.replace("_", " ").lower()]
                found = None
                for key in key_options:
                    if key in class_index:
                        found = int(class_index[key])
                        break
                if found is None:
                    raise ValueError(
                        f"Could not map ImageFolder class '{c}' to an ImageNet model index. "
                        "Provide a complete --imagenet_class_index_json."
                    )
                target_remap[int(old_idx)] = int(found)
        else:
            raise ValueError(
                "ImageNet ImageFolder class names are not numeric and no --imagenet_class_index_json was provided. "
                "Use torchvision.datasets.ImageNet root with devkit metadata, numeric 0..999 folders, or provide "
                "the common imagenet_class_index.json mapping."
            )
        return RemappedTargetDataset(ds_folder, target_remap)

    raise RuntimeError(
        "Could not load ImageNet. Provide --data_root pointing to a torchvision ImageNet root, "
        "or --imagenet_val_dir pointing to an ImageFolder-style validation directory."
    )



def load_imagenet_cache(cache_path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load a persistent ImageNet probability cache.

    Expected NPZ keys:
      * probs:  shape (N,1000), model probabilities;
      * labels: shape (N,), integer ImageNet class indices;
      * meta_json: optional JSON metadata string.

    The loader also accepts the alias key `probabilities` for `probs`.
    """
    if not cache_path:
        raise ValueError("Empty ImageNet cache path.")
    path = Path(cache_path)
    if not path.exists():
        raise FileNotFoundError(f"ImageNet cache not found: {cache_path}")
    obj = np.load(path, allow_pickle=False)
    if "probs" in obj:
        probs = obj["probs"]
    elif "probabilities" in obj:
        probs = obj["probabilities"]
    else:
        raise KeyError("ImageNet cache must contain key 'probs' or 'probabilities'.")
    if "labels" not in obj:
        raise KeyError("ImageNet cache must contain key 'labels'.")
    labels = obj["labels"].astype(int).reshape(-1)
    probs = probs.astype(np.float32, copy=False)
    if probs.ndim != 2:
        raise ValueError(f"Cached probabilities must be 2-D, got shape {probs.shape}.")
    if len(labels) != probs.shape[0]:
        raise ValueError(f"Cached labels/probs length mismatch: {len(labels)} vs {probs.shape[0]}.")
    meta: Dict[str, Any] = {}
    if "meta_json" in obj:
        raw = obj["meta_json"]
        try:
            meta_str = str(raw.item() if getattr(raw, "shape", ()) == () else raw)
            meta = json.loads(meta_str)
        except Exception:
            meta = {"meta_json_parse_error": True}
    return labels, probs, meta


def cache_imagenet_probabilities(args: argparse.Namespace) -> Path:
    """Precompute ImageNet validation labels and probabilities once.

    This is intentionally separate from `run`: ImageNet image decoding and
    ResNet-50 inference dominate runtime. A cache lets all conformal methods and
    all MC splits operate on arrays only.
    """
    import torch
    from torch.utils.data import DataLoader
    import torchvision.models as models

    model_id = str(args.imagenet_model).lower()
    if model_id != "resnet50":
        raise ValueError("Only --imagenet_model resnet50 is currently supported.")
    weights = models.ResNet50_Weights.DEFAULT
    transform = weights.transforms()

    ds = load_imagenet_dataset(
        data_root=str(args.data_root),
        val_dir=str(args.imagenet_val_dir or ""),
        transform=transform,
        class_index_json=str(args.imagenet_class_index_json or ""),
    )

    device = str(args.device).lower()
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is not available.")

    model = models.resnet50(weights=weights).to(device).eval()
    loader = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory and device == "cuda"),
        persistent_workers=bool(int(args.num_workers) > 0),
    )

    probs_parts: List[np.ndarray] = []
    labels_parts: List[np.ndarray] = []
    t0 = time.time()
    total = len(ds)
    seen = 0
    with torch.inference_mode():
        for batch_idx, (x, y) in enumerate(loader):
            if device == "cuda":
                x = x.to(device, non_blocking=True)
            else:
                x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float32)
            probs_parts.append(probs)
            labels_parts.append(y.detach().cpu().numpy().astype(np.int64))
            seen += int(len(y))
            if not bool(args.quiet) and (batch_idx == 0 or seen == total or batch_idx % int(args.log_every) == 0):
                elapsed = max(time.time() - t0, 1e-9)
                print(f"[cache_imagenet] {seen}/{total} images | {seen/elapsed:.2f} img/s")

    probs_all = np.concatenate(probs_parts, axis=0)
    labels_all = np.concatenate(labels_parts, axis=0)
    meta = {
        "dataset": "imagenet",
        "split": "val",
        "model": model_id,
        "num_samples": int(len(labels_all)),
        "num_classes": int(probs_all.shape[1]),
        "data_root": str(args.data_root),
        "imagenet_val_dir": str(args.imagenet_val_dir or ""),
        "class_index_json": str(args.imagenet_class_index_json or ""),
        "created_at": datetime.now().isoformat(),
        "torch_device": device,
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
    }

    out_path = Path(args.cache_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {
        "probs": probs_all.astype(np.float32, copy=False),
        "labels": labels_all.astype(np.int64, copy=False),
        "meta_json": np.array(json.dumps(meta)),
    }
    if bool(args.compressed):
        np.savez_compressed(out_path, **save_kwargs)
    else:
        np.savez(out_path, **save_kwargs)
    if not bool(args.quiet):
        mb = out_path.stat().st_size / (1024 ** 2)
        print(f"Saved ImageNet cache: {out_path} ({mb:.1f} MiB)")
    return out_path

def make_label_partitions(n_classes: int, K: int) -> List[set]:
    chunks = np.array_split(np.arange(n_classes), K)
    return [set(int(v) for v in c.tolist()) for c in chunks]


def label_to_client_map(client_label_sets: List[set]) -> Dict[int, int]:
    m: Dict[int, int] = {}
    for k, ss in enumerate(client_label_sets):
        for y in ss:
            m[int(y)] = int(k)
    return m


def make_dirichlet_client_indices(
    labels: np.ndarray,
    K: int,
    beta: float = 0.3,
    min_count: int = 5,
    seed: int = 0,
    max_retries: int = 2000,
) -> List[np.ndarray]:
    """Sample-level non-i.i.d. client split via class-wise Dirichlet allocation.

    This is used especially for CIFAR-10 with K > number of classes. A
    class-disjoint split would create empty clients, which makes the
    FedCP/GC-FCP finite-sample correction degenerate. The Dirichlet split
    preserves non-i.i.d. label skew while ensuring every client has at least
    `min_count` calibration samples.
    """
    labels = np.asarray(labels, dtype=int)
    n = int(len(labels))
    K = int(K)
    if K <= 0:
        raise ValueError("K must be positive.")
    if n == 0:
        return [np.array([], dtype=int) for _ in range(K)]

    min_count = int(max(0, min_count))
    if min_count * K > n:
        min_count = max(1, n // K)

    beta = float(beta)
    if beta <= 0:
        raise ValueError("Dirichlet beta must be positive.")

    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    best: Optional[List[np.ndarray]] = None
    best_min = -1

    for _ in range(int(max_retries)):
        client_indices: List[List[int]] = [[] for _ in range(K)]
        for y in classes:
            idx_y = np.where(labels == int(y))[0].astype(int)
            rng.shuffle(idx_y)
            proportions = rng.dirichlet(beta * np.ones(K, dtype=float))
            cuts = (np.cumsum(proportions)[:-1] * len(idx_y)).astype(int)
            parts = np.split(idx_y, cuts)
            for k, part in enumerate(parts):
                if len(part):
                    client_indices[k].extend(int(v) for v in part.tolist())

        out = [np.array(sorted(v), dtype=int) for v in client_indices]
        sizes = np.array([len(v) for v in out], dtype=int)
        if int(sizes.min()) > best_min:
            best = out
            best_min = int(sizes.min())
        if int(sizes.min()) >= min_count:
            return out

    if best is None:
        raise RuntimeError("Could not create a Dirichlet client split.")
    if best_min <= 0:
        raise RuntimeError(
            "Could not create a non-empty Dirichlet split. Try increasing n_cal, "
            "decreasing --clients, increasing --dirichlet_beta, or lowering --min_client_count."
        )
    return best


def make_label_partition_client_indices(labels: np.ndarray, n_classes: int, K: int) -> Tuple[List[np.ndarray], List[set]]:
    """Class-disjoint split used by the original scripts."""
    client_label_sets = make_label_partitions(n_classes, K)
    label_map = label_to_client_map(client_label_sets)
    labels = np.asarray(labels, dtype=int)
    client_indices: List[np.ndarray] = []
    for k in range(int(K)):
        mask = np.array([label_map.get(int(y), K - 1) == k for y in labels], dtype=bool)
        client_indices.append(np.where(mask)[0].astype(int))
    return client_indices, client_label_sets


def fedcp_mass_diagnostic(n_k: Sequence[int], pi: Sequence[float], alpha: float) -> Dict[str, Any]:
    """Finite calibration/virtual mass diagnostics for FedCP-style weights."""
    n_k_arr = np.asarray(n_k, dtype=float)
    pi_arr = np.asarray(pi, dtype=float)
    if pi_arr.sum() > 0:
        pi_arr = pi_arr / pi_arr.sum()
    finite_mass = float(np.sum(pi_arr * n_k_arr / (n_k_arr + 1.0))) if len(n_k_arr) else 0.0
    virtual_mass = float(np.sum(pi_arr / (n_k_arr + 1.0))) if len(n_k_arr) else 0.0
    return {
        "empty_clients": int(np.sum(n_k_arr == 0)),
        "min_nk": int(np.min(n_k_arr)) if len(n_k_arr) else 0,
        "median_nk": float(np.median(n_k_arr)) if len(n_k_arr) else 0.0,
        "max_nk": int(np.max(n_k_arr)) if len(n_k_arr) else 0,
        "finite_mass": finite_mass,
        "virtual_mass": virtual_mass,
        "virtual_mass_exceeds_alpha": bool(virtual_mass > float(alpha)),
    }


# -------------------------
# Score functions
# -------------------------

def classification_score_matrix(
    probs: np.ndarray,
    score: str,
    raps_lambda: float = 0.01,
    raps_k_reg: int = 5,
) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    score = score.lower()
    n, c = probs.shape

    if score in {"thr", "hps"}:
        return 1.0 - probs

    order = np.argsort(-probs, axis=1)
    sorted_probs = np.take_along_axis(probs, order, axis=1)
    cum = np.cumsum(sorted_probs, axis=1)
    ranks = np.empty_like(order)
    ranks[np.arange(n)[:, None], order] = np.arange(c)[None, :]
    aps = np.empty_like(probs)
    aps[np.arange(n)[:, None], order] = cum

    if score == "aps":
        return aps

    if score in {"raps", "rapr"}:
        # rank is 0-based; regularization starts after k_reg labels.
        penalty = raps_lambda * np.maximum(ranks + 1 - int(raps_k_reg), 0)
        return aps + penalty

    raise ValueError(f"Unknown classification score: {score}")


# -------------------------
# Synthetic data
# -------------------------

def synthetic_generate_x(mu: float, sigma: float, size: int, rng: np.random.Generator, bounds=(0.0, 5.0)) -> np.ndarray:
    a = (bounds[0] - mu) / sigma
    b = (bounds[1] - mu) / sigma
    return truncnorm.rvs(a, b, loc=mu, scale=sigma, size=size, random_state=rng)


def synthetic_generate_y(x: np.ndarray, client_index_1based: int, rng: np.random.Generator, label_shift: bool = True) -> np.ndarray:
    pois = rng.poisson(np.sin(x) ** 2 + 0.1, size=len(x))
    eps1 = rng.normal(0.0, 1.0, size=len(x))
    U = rng.uniform(0.0, 1.0, size=len(x))
    eps2 = rng.normal(0.0, 1.0, size=len(x))
    normal = rng.normal(0.0, 0.01 * client_index_1based ** 2, size=len(x)) if label_shift else 0.0
    return pois + 0.03 * x * eps1 + 25.0 * (U < 0.01) * eps2 + normal


def make_synthetic_data(
    seed: int,
    alpha: float,
    K: int,
    pi: np.ndarray,
    n_cal: int,
    n_test: int,
    synthetic_train: int,
    poly_degree: int,
    label_shift: bool = True,
) -> ExperimentData:
    rng = np.random.default_rng(seed)
    bounds = (0.0, 5.0)
    n_k = np.array([n_cal // K] * K, dtype=int)
    n_k[: n_cal % K] += 1

    poly = PolynomialFeatures(poly_degree)

    # Training data used only to fit the base predictor.
    x_train_parts = []
    y_train_parts = []
    train_per_client = max(20, synthetic_train // K)
    for k in range(K):
        mu = 0.5 + 4.0 * k / max(K - 1, 1)
        sigma = 0.5 + 0.1 * k
        xk = synthetic_generate_x(mu, sigma, train_per_client, rng, bounds)
        yk = synthetic_generate_y(xk, k + 1, rng, label_shift)
        x_train_parts.append(xk)
        y_train_parts.append(yk)

    x_train = np.concatenate(x_train_parts)
    y_train = np.concatenate(y_train_parts)
    reg = LinearRegression().fit(poly.fit_transform(x_train.reshape(-1, 1)), y_train)

    groups = [(0.0, 2.0), (1.0, 3.0), (2.0, 4.0), (3.0, 5.0)]
    group_names = ["[0,2]", "[1,3]", "[2,4]", "[3,5]"]

    client_scores: List[np.ndarray] = []
    client_phi: List[np.ndarray] = []
    client_x: List[np.ndarray] = []
    for k in range(K):
        mu = 0.5 + 4.0 * k / max(K - 1, 1)
        sigma = 0.5 + 0.1 * k
        xk = synthetic_generate_x(mu, sigma, int(n_k[k]), rng, bounds)
        yk = synthetic_generate_y(xk, k + 1, rng, label_shift)
        yhat = reg.predict(poly.transform(xk.reshape(-1, 1)))
        scores = np.abs(yk - yhat)
        client_scores.append(scores.astype(float))
        client_phi.append(phi_from_intervals(xk, groups))
        client_x.append(xk.astype(float))

    x_test_parts = []
    y_test_parts = []
    client_choices = rng.choice(np.arange(K), size=n_test, p=pi)
    for k in range(K):
        nk = int(np.sum(client_choices == k))
        if nk == 0:
            continue
        mu = 0.5 + 4.0 * k / max(K - 1, 1)
        sigma = 0.5 + 0.1 * k
        xk = synthetic_generate_x(mu, sigma, nk, rng, bounds)
        yk = synthetic_generate_y(xk, k + 1, rng, label_shift)
        x_test_parts.append(xk)
        y_test_parts.append(yk)
    x_test = np.concatenate(x_test_parts)
    y_test = np.concatenate(y_test_parts)
    perm = rng.permutation(len(x_test))
    x_test = x_test[perm]
    y_test = y_test[perm]
    yhat_test = reg.predict(poly.transform(x_test.reshape(-1, 1)))
    test_scores = np.abs(y_test - yhat_test)

    score_upper = float(max(np.max(test_scores), max(np.max(s) for s in client_scores)) * 1.25 + 1e-8)

    return ExperimentData(
        dataset="synthetic",
        score="reg",
        alpha=alpha,
        group_names=group_names,
        client_scores=client_scores,
        client_phi=client_phi,
        client_labels=None,
        client_x_scalar=client_x,
        pi=pi,
        test_scores_true=test_scores.astype(float),
        test_scores_all=None,
        test_phi=phi_from_intervals(x_test, groups),
        test_labels=None,
        set_kind="regression",
        score_upper=score_upper,
        metadata={
            "groups": groups,
            "poly_degree": poly_degree,
            "label_shift": label_shift,
            "n_k": n_k.tolist(),
        },
    )


# -------------------------
# Real data
# -------------------------

_CLASSIFIER_MODEL_CACHE: Dict[Tuple[str, str, str], Any] = {}
_REAL_PROBS_CACHE: Dict[Tuple[Any, ...], Dict[str, Any]] = {}


def load_classifier_model(dataset: str, device: str, imagenet_model: str = "resnet50"):
    import torch
    import torch.hub

    dataset = str(dataset).lower()
    model_id = str(imagenet_model).lower() if dataset == "imagenet" else dataset
    key = (dataset, str(device), model_id)
    if key in _CLASSIFIER_MODEL_CACHE:
        return _CLASSIFIER_MODEL_CACHE[key]

    if dataset in {"cifar10", "cifar100"}:
        torch.hub.set_dir("./model")
        model_name = "cifar10_resnet56" if dataset == "cifar10" else "cifar100_resnet56"
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", model_name, pretrained=True)
    elif dataset == "imagenet":
        import torchvision.models as models
        if model_id != "resnet50":
            raise ValueError("Only --imagenet_model resnet50 is currently supported.")
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
    else:
        raise ValueError(dataset)

    model.to(device).eval()
    _CLASSIFIER_MODEL_CACHE[key] = model
    return model


def load_cifar_model(dataset: str, device: str):
    return load_classifier_model(dataset, device)


def compute_probs_torch(model, images, labels, dataset: str, device: str, batch_size: int = 128) -> np.ndarray:
    import torch
    import torchvision.transforms as transforms

    if dataset in {"cifar10", "cifar100"}:
        normalize = transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761],
        )
    else:
        normalize = lambda x: x

    probs: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch_x = torch.stack(images[i : i + batch_size]).to(device)
            batch_x = normalize(batch_x)
            logits = model(batch_x)
            soft = torch.softmax(logits, dim=1)
            probs.append(soft.cpu().numpy())
    return np.concatenate(probs, axis=0)


def make_real_data(
    seed: int,
    dataset: str,
    score: str,
    alpha: float,
    K: int,
    pi: np.ndarray,
    n_cal: int,
    n_test: int,
    group_mode: str,
    top_r: int,
    raps_lambda: float,
    raps_k_reg: int,
    batch_size: int,
    x_feature: str = "confidence",
    client_split: str = "auto",
    dirichlet_beta: float = 0.3,
    min_client_count: int = 5,
    data_root: str = "./data",
    imagenet_val_dir: str = "",
    imagenet_class_index_json: str = "",
    imagenet_meta_map: str = "",
    imagenet_model: str = "resnet50",
    imagenet_cache: str = "",
    ambiguity_score: str = "margin",
    ambiguity_bins: int = 5,
    ambiguity_binning: str = "quantile",
) -> ExperimentData:
    import torch
    import torchvision
    import torchvision.transforms as transforms

    rng = np.random.default_rng(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = dataset.lower()
    x_feature = (x_feature or "confidence").lower()
    if x_feature not in {"pred_label", "confidence", "entropy"}:
        raise ValueError("x_feature must be one of {'pred_label', 'confidence', 'entropy'}")

    data_root_path = Path(data_root)
    use_imagenet_cache = bool(dataset == "imagenet" and str(imagenet_cache or "").strip())
    cache_meta: Dict[str, Any] = {}

    if use_imagenet_cache:
        all_labels, all_probs, cache_meta = load_imagenet_cache(str(imagenet_cache))
        n_classes = int(all_probs.shape[1])
        if n_classes != 1000:
            raise ValueError(f"Expected ImageNet cache with 1000 classes, got {n_classes}.")
        N = int(len(all_labels))
        ds = None
    else:
        if dataset in {"cifar10", "cifar100"}:
            transform = transforms.ToTensor()
            if dataset == "cifar10":
                ds = torchvision.datasets.CIFAR10(root=str(data_root_path), train=False, download=True, transform=transform)
                n_classes = 10
            else:
                ds = torchvision.datasets.CIFAR100(root=str(data_root_path), train=False, download=True, transform=transform)
                n_classes = 100
        elif dataset == "imagenet":
            import torchvision.models as models
            if str(imagenet_model).lower() != "resnet50":
                raise ValueError("Only --imagenet_model resnet50 is currently supported.")
            weights = models.ResNet50_Weights.DEFAULT
            transform = weights.transforms()
            ds = load_imagenet_dataset(
                data_root=str(data_root_path),
                val_dir=str(imagenet_val_dir or ""),
                transform=transform,
                class_index_json=str(imagenet_class_index_json or ""),
            )
            n_classes = 1000
        else:
            raise ValueError(f"Unsupported real dataset: {dataset}")
        N = len(ds)

    total_needed = min(N, n_cal + n_test)
    indices = rng.permutation(N)[:total_needed]
    if total_needed < n_cal + n_test:
        n_cal = min(n_cal, total_needed // 2)
        n_test = total_needed - n_cal
    cal_idx = indices[:n_cal]
    test_idx = indices[n_cal : n_cal + n_test]

    if use_imagenet_cache:
        cal_labels_all = all_labels[cal_idx].astype(int)
        test_labels = all_labels[test_idx].astype(int)
        probs_cal = all_probs[cal_idx].astype(np.float32, copy=False)
        probs_test = all_probs[test_idx].astype(np.float32, copy=False)
    else:
        # Cache labels and model probabilities across score functions within the
        # same Monte Carlo split. This avoids re-loading the model and re-running
        # inference separately for THR, APS, and RAPS. This in-process cache is
        # useful for CIFAR and for ImageNet smoke tests; for real ImageNet runs,
        # prefer --imagenet_cache created by the cache_imagenet command.
        prob_key = (
            dataset,
            str(data_root_path),
            str(imagenet_val_dir or "") if dataset == "imagenet" else "",
            str(imagenet_class_index_json or "") if dataset == "imagenet" else "",
            str(imagenet_model or "") if dataset == "imagenet" else "",
            tuple(int(i) for i in cal_idx.tolist()),
            tuple(int(i) for i in test_idx.tolist()),
            int(batch_size),
            str(device),
        )
        cached = _REAL_PROBS_CACHE.get(prob_key)
        if cached is None:
            cal_items = [ds[int(i)] for i in cal_idx]
            test_items = [ds[int(i)] for i in test_idx]
            cal_images = [x for x, _ in cal_items]
            cal_labels_all = np.array([int(y) for _, y in cal_items], dtype=int)
            test_images = [x for x, _ in test_items]
            test_labels = np.array([int(y) for _, y in test_items], dtype=int)

            model = load_classifier_model(dataset, device, imagenet_model=imagenet_model)
            probs_cal = compute_probs_torch(model, cal_images, cal_labels_all, dataset, device, batch_size)
            probs_test = compute_probs_torch(model, test_images, test_labels, dataset, device, batch_size)
            _REAL_PROBS_CACHE[prob_key] = {
                "cal_labels_all": cal_labels_all,
                "test_labels": test_labels,
                "probs_cal": probs_cal,
                "probs_test": probs_test,
            }
        else:
            cal_labels_all = cached["cal_labels_all"]
            test_labels = cached["test_labels"]
            probs_cal = cached["probs_cal"]
            probs_test = cached["probs_test"]

    score_matrix_cal = classification_score_matrix(probs_cal, score, raps_lambda, raps_k_reg)
    score_matrix_test = classification_score_matrix(probs_test, score, raps_lambda, raps_k_reg)
    score_cal_true = score_matrix_cal[np.arange(len(cal_labels_all)), cal_labels_all]
    score_test_true = score_matrix_test[np.arange(len(test_labels)), test_labels]

    imagenet_fine_to_meta: Optional[np.ndarray] = None
    if dataset == "cifar100" and group_mode == "coarse_topr":
        phi_cal_all = phi_cifar100_topr(probs_cal, top_r=top_r)
        phi_test = phi_cifar100_topr(probs_test, top_r=top_r)
        group_names = CIFAR100_COARSE_NAMES[:]
        group_meta: Any = {"group_mode": group_mode, "top_r": top_r, "coarse_names": group_names}
    elif dataset == "imagenet" and group_mode in {"coarse_topr", "imagenet_meta_topr"}:
        categories = get_imagenet_categories(imagenet_model)
        imagenet_fine_to_meta, group_names, meta_source = build_imagenet_meta_mapping(categories, imagenet_meta_map)
        phi_cal_all = phi_imagenet_meta_topr(probs_cal, imagenet_fine_to_meta, top_r=top_r)
        phi_test = phi_imagenet_meta_topr(probs_test, imagenet_fine_to_meta, top_r=top_r)
        group_meta = {
            "group_mode": "imagenet_meta_topr",
            "top_r": int(top_r),
            "meta_group_names": group_names,
            "imagenet_model": str(imagenet_model),
            "imagenet_meta_mapping_source": meta_source,
            "imagenet_meta_class_counts": np.bincount(imagenet_fine_to_meta, minlength=10).astype(int).tolist(),
        }
    elif dataset == "imagenet" and group_mode in {"imagenet_semantic_ambiguity", "imagenet_semantic_margin"}:
        categories = get_imagenet_categories(imagenet_model)
        imagenet_fine_to_meta, meta_group_names, meta_source = build_imagenet_meta_mapping(categories, imagenet_meta_map)
        phi_cal_all, phi_test, group_names, group_meta_extra = phi_imagenet_semantic_ambiguity(
            probs_cal=probs_cal,
            probs_test=probs_test,
            fine_to_meta=imagenet_fine_to_meta,
            meta_names=meta_group_names,
            num_bins=int(ambiguity_bins),
            ambiguity_score=str(ambiguity_score),
            ambiguity_binning=str(ambiguity_binning),
        )
        group_meta = {
            **group_meta_extra,
            "group_mode_alias": str(group_mode),
            "imagenet_model": str(imagenet_model),
            "imagenet_meta_mapping_source": meta_source,
            "imagenet_meta_class_counts": np.bincount(imagenet_fine_to_meta, minlength=10).astype(int).tolist(),
        }
    else:
        effective_group_mode = "pred_window" if group_mode == "coarse_topr" else group_mode

        if dataset == "cifar10" and x_feature in {"confidence", "entropy"}:
            # CIFAR-10 group design from realdata_additional.py: form groups on
            # a scalar prediction feature rather than the paper's predicted-label
            # windows. This gives difficulty-style overlapping groups.
            if x_feature == "confidence":
                x_min, x_max = 0.5, 1.01
                x_cal = np.max(probs_cal, axis=1).astype(float)
                x_test = np.max(probs_test, axis=1).astype(float)
            else:
                x_min, x_max = 0.0, float(np.log(n_classes) + 1e-6)
                x_cal = (-np.sum(probs_cal * np.log(probs_cal + 1e-12), axis=1)).astype(float)
                x_test = (-np.sum(probs_test * np.log(probs_test + 1e-12), axis=1)).astype(float)

            a, b = float(x_min), float(x_max)
            if effective_group_mode == "disjoint_pred":
                edges = np.linspace(a, b, num=6)
                intervals = [(float(edges[i]), float(edges[i + 1])) for i in range(5)]
            else:
                intervals = [
                    (a + 0.00 * (b - a), a + 0.55 * (b - a)),
                    (a + 0.35 * (b - a), a + 0.75 * (b - a)),
                    (a + 0.55 * (b - a), a + 0.90 * (b - a)),
                    (a + 0.75 * (b - a), b),
                ]
            phi_cal_all = phi_from_intervals(x_cal, intervals)
            phi_test = phi_from_intervals(x_test, intervals)
            group_names = [f"G{i+1}" for i in range(phi_cal_all.shape[1])]
            group_meta = {
                "group_mode": effective_group_mode,
                "x_feature": x_feature,
                "intervals": intervals,
                "source": "realdata_additional.py difficulty-style CIFAR-10 grouping",
            }
        else:
            # Predicted-label windows. For CIFAR-10 this is retained only via
            # --x_feature pred_label, which recovers the paper-style setting.
            pred_cal = np.argmax(probs_cal, axis=1).astype(float)
            pred_test = np.argmax(probs_test, axis=1).astype(float)
            if dataset == "cifar10":
                if effective_group_mode == "disjoint_pred":
                    intervals = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)]
                else:
                    intervals = [(0, 4), (2, 6), (4, 8), (6, 10)]
            else:
                # CIFAR-100 fallback based on predicted fine-label windows.
                width = 20
                step = 10 if group_mode != "disjoint_pred" else 20
                intervals = []
                lo = 0
                while lo < n_classes:
                    hi = min(n_classes, lo + width)
                    intervals.append((lo, hi))
                    lo += step
                    if group_mode == "disjoint_pred" and lo >= n_classes:
                        break
                if intervals[-1][1] < n_classes:
                    intervals[-1] = (intervals[-1][0], n_classes)
            phi_cal_all = phi_from_intervals(pred_cal, intervals)
            phi_test = phi_from_intervals(pred_test, intervals)
            group_names = [f"G{i+1}" for i in range(phi_cal_all.shape[1])]
            group_meta = {"group_mode": effective_group_mode, "x_feature": "pred_label", "intervals": intervals}


    # Client allocation. The original scripts used class-disjoint label
    # partitions. For CIFAR-10 with K > 10 this creates empty clients, which
    # makes FedCP/GC-FCP return trivial full-label sets. In auto mode, CIFAR-10
    # switches to Dirichlet sample-level allocation when K exceeds the number of
    # classes. Users can also force Dirichlet with --client_split dirichlet.
    split_req = (client_split or "auto").lower()
    if split_req not in {"auto", "label", "dirichlet"}:
        raise ValueError("client_split must be one of {'auto','label','dirichlet'}")
    if split_req == "auto":
        split_used = "dirichlet" if (dataset == "imagenet" or (dataset == "cifar10" and K > n_classes)) else "label"
    else:
        split_used = split_req

    if split_used == "dirichlet":
        client_indices = make_dirichlet_client_indices(
            cal_labels_all,
            K=K,
            beta=float(dirichlet_beta),
            min_count=int(min_client_count),
            seed=int(seed) + 271828,
        )
        client_label_sets = []
        for idx_k in client_indices:
            client_label_sets.append(set(int(v) for v in np.unique(cal_labels_all[idx_k]).tolist()))
    else:
        client_indices, client_label_sets = make_label_partition_client_indices(cal_labels_all, n_classes, K)

    client_scores: List[np.ndarray] = []
    client_phi: List[np.ndarray] = []
    client_labels: List[np.ndarray] = []
    client_x: List[np.ndarray] = []

    if dataset == "cifar10" and x_feature == "confidence":
        pred_or_scalar = np.max(probs_cal, axis=1).astype(float)
    elif dataset == "cifar10" and x_feature == "entropy":
        pred_or_scalar = (-np.sum(probs_cal * np.log(probs_cal + 1e-12), axis=1)).astype(float)
    else:
        pred_or_scalar = np.argmax(probs_cal, axis=1).astype(float)
    if dataset == "cifar100" and group_mode == "coarse_topr":
        # A scalar is only used for optional density-ratio diagnostics; use top coarse class.
        coarse = np.zeros((probs_cal.shape[0], 20), dtype=float)
        for fine in range(100):
            coarse[:, CIFAR100_FINE_TO_COARSE[fine]] += probs_cal[:, fine]
        pred_or_scalar = np.argmax(coarse, axis=1).astype(float)
    elif dataset == "imagenet" and group_mode in {"coarse_topr", "imagenet_meta_topr", "imagenet_semantic_ambiguity", "imagenet_semantic_margin"}:
        # A scalar is only used for optional density-ratio diagnostics; use top meta group.
        if imagenet_fine_to_meta is None:
            categories = get_imagenet_categories(imagenet_model)
            imagenet_fine_to_meta, _, _ = build_imagenet_meta_mapping(categories, imagenet_meta_map)
        meta_probs = imagenet_meta_probabilities(probs_cal, imagenet_fine_to_meta, n_meta=10)
        pred_or_scalar = np.argmax(meta_probs, axis=1).astype(float)

    for k in range(K):
        idx = np.asarray(client_indices[k], dtype=int)
        client_scores.append(score_cal_true[idx].astype(float))
        client_phi.append(phi_cal_all[idx].astype(int))
        client_labels.append(cal_labels_all[idx].astype(int))
        client_x.append(pred_or_scalar[idx].astype(float))

    n_k_diag = np.array([len(s) for s in client_scores], dtype=int)
    mass_diag = fedcp_mass_diagnostic(n_k_diag, pi, alpha)

    score_upper = float(np.nanmax([np.max(score_matrix_test), np.max(score_matrix_cal)]) * 1.05 + 1e-8)
    if not np.isfinite(score_upper) or score_upper <= 0:
        score_upper = 1.0

    return ExperimentData(
        dataset=dataset,
        score=score.lower(),
        alpha=alpha,
        group_names=group_names,
        client_scores=client_scores,
        client_phi=client_phi,
        client_labels=client_labels,
        client_x_scalar=client_x,
        pi=pi,
        test_scores_true=score_test_true.astype(float),
        test_scores_all=score_matrix_test.astype(float),
        test_phi=phi_test.astype(int),
        test_labels=test_labels,
        set_kind="classification",
        score_upper=score_upper,
        metadata={
            "n_classes": n_classes,
            "client_split_requested": split_req,
            "client_split_used": split_used,
            "dirichlet_beta": float(dirichlet_beta) if split_used == "dirichlet" else None,
            "min_client_count": int(min_client_count) if split_used == "dirichlet" else None,
            "client_label_sets": [sorted(list(s)) for s in client_label_sets],
            "n_k": n_k_diag.tolist(),
            "fedcp_mass_diagnostic": mass_diag,
            "imagenet_cache": str(imagenet_cache or "") if dataset == "imagenet" else "",
            "imagenet_cache_meta": cache_meta if (dataset == "imagenet" and cache_meta) else {},
            **group_meta,
        },
    )


# -------------------------
# Evaluation and methods
# -------------------------

def flatten_clients(data: ExperimentData) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Return scores, phi, weights, labels, client_ids."""
    n_k = np.array([len(s) for s in data.client_scores], dtype=int)
    weights_by_client = [np.full(n_k[k], data.pi[k] / (n_k[k] + 1.0), dtype=float) for k in range(data.K)]
    scores = np.concatenate(data.client_scores) if data.n_cal > 0 else np.array([], dtype=float)
    phi = np.vstack(data.client_phi) if data.n_cal > 0 else np.zeros((0, data.n_groups), dtype=int)
    weights = np.concatenate(weights_by_client) if data.n_cal > 0 else np.array([], dtype=float)
    labels = None
    if data.client_labels is not None:
        labels = np.concatenate(data.client_labels) if data.n_cal > 0 else np.array([], dtype=int)
    client_ids = np.concatenate([np.full(n_k[k], k, dtype=int) for k in range(data.K)]) if data.n_cal > 0 else np.array([], dtype=int)
    return scores, phi, weights, labels, client_ids


def evaluate_taus(
    data: ExperimentData,
    taus: np.ndarray,
) -> Tuple[float, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate marginal and group-level coverage and set size.

    Returns:
      marg_cov: scalar marginal coverage;
      group_cov: length-|G| vector of group coverages;
      avg_set_size: scalar marginal average set size / interval length;
      set_sizes: per-test-point set sizes;
      group_size: length-|G| vector of group test counts;
      group_avg_set_size: length-|G| vector of per-group average set sizes.

    `group_size` is the denominator used for pooled group coverage.
    `group_avg_set_size` is recorded separately in raw_results.csv so that
    per-group efficiency can be inspected just like per-group coverage.
    """
    taus = np.asarray(taus, dtype=float)
    if taus.ndim == 0:
        taus = np.full(data.n_test, float(taus), dtype=float)
    cover = data.test_scores_true <= taus + 1e-12

    if data.set_kind == "classification":
        assert data.test_scores_all is not None
        set_sizes = np.sum(data.test_scores_all <= taus[:, None] + 1e-12, axis=1).astype(float)
    else:
        set_sizes = 2.0 * taus
        set_sizes[~np.isfinite(set_sizes)] = float("inf")

    marg_cov = float(np.mean(cover)) if len(cover) else float("nan")
    avg_set_size = float(np.mean(set_sizes)) if len(set_sizes) else float("nan")

    group_cov = np.full(data.n_groups, np.nan, dtype=float)
    group_size = np.zeros(data.n_groups, dtype=int)
    group_avg_set_size = np.full(data.n_groups, np.nan, dtype=float)
    for g in range(data.n_groups):
        idx = data.test_phi[:, g] > 0
        group_size[g] = int(np.sum(idx))
        if group_size[g] > 0:
            group_cov[g] = float(np.mean(cover[idx]))
            group_avg_set_size[g] = float(np.mean(set_sizes[idx]))

    return marg_cov, group_cov, avg_set_size, set_sizes, group_size, group_avg_set_size


def make_record(
    data: ExperimentData,
    seed: int,
    mc: int,
    method_key: str,
    method: str,
    taus: np.ndarray,
    atoms: int,
    coreset: int,
    comm: int,
    comp_speedup: float,
    runtime_sec: float,
    details: Dict[str, Any],
) -> MethodRecord:
    marg_cov, group_cov, avg_set_size, _, group_sizes, group_avg_set_size = evaluate_taus(data, taus)
    n_classes_meta = int(data.metadata.get("n_classes", 0) or 0)
    norm_avg_set_size = float(avg_set_size / n_classes_meta) if (data.set_kind == "classification" and n_classes_meta > 0 and np.isfinite(avg_set_size)) else float("nan")
    if data.set_kind == "classification" and n_classes_meta > 0:
        group_norm_avg_set_size = group_avg_set_size / float(n_classes_meta)
    else:
        group_norm_avg_set_size = np.full_like(group_avg_set_size, np.nan, dtype=float)
    valid = np.isfinite(group_cov)
    if np.any(valid):
        worst = float(np.nanmin(group_cov))
        avg_g = float(np.nanmean(group_cov))
        max_gap = float(np.nanmax(np.abs(group_cov[valid] - (1.0 - data.alpha))))
    else:
        worst, avg_g, max_gap = float("nan"), float("nan"), float("nan")

    return MethodRecord(
        seed=int(seed),
        mc=int(mc),
        dataset=data.dataset,
        score=data.score,
        method_key=method_key,
        method=method,
        alpha=float(data.alpha),
        marg_cov=marg_cov,
        worst_group_cov=worst,
        avg_group_cov=avg_g,
        max_group_gap=max_gap,
        avg_set_size=avg_set_size,
        norm_avg_set_size=norm_avg_set_size,
        atoms=int(atoms),
        coreset=int(coreset),
        comm=int(comm),
        comp_speedup=float(comp_speedup) if comp_speedup is not None else float("nan"),
        n_cal=int(data.n_cal),
        n_test=int(data.n_test),
        n_groups=int(data.n_groups),
        runtime_sec=float(runtime_sec),
        group_cov_json=json.dumps(group_cov.tolist()),
        group_size_json=json.dumps(group_sizes.tolist()),
        group_avg_set_size_json=json.dumps(group_avg_set_size.tolist()),
        group_norm_avg_set_size_json=json.dumps(group_norm_avg_set_size.tolist()),
        details_json=json.dumps(details, sort_keys=True),
    )


def solve_lp_threshold_for_phi(
    phi_test: np.ndarray,
    phi_all: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
    w_test: float,
    alpha: float,
    score_upper: float,
    max_iter: int = 40,
) -> float:
    """Dual LP bisection used for CondCP / GC-FCP."""
    scores = np.asarray(scores, dtype=float)
    weights = np.asarray(weights, dtype=float)
    phi_all = np.asarray(phi_all, dtype=float)
    phi_test = np.asarray(phi_test, dtype=float)

    if len(scores) == 0:
        return float("inf")

    active_cols = np.where((np.sum(np.abs(phi_all), axis=0) + np.abs(phi_test)) > 0)[0]
    if len(active_cols) == 0:
        return weighted_quantile_target(scores, weights, (1.0 - alpha) * np.sum(weights))

    phi_all_r = phi_all[:, active_cols]
    phi_test_r = phi_test[active_cols]

    A_eq = np.hstack((phi_all_r.T, phi_test_r.reshape(-1, 1)))
    b_eq = np.zeros(len(active_cols), dtype=float)
    bounds = [(-float(w) * alpha, float(w) * (1.0 - alpha)) for w in weights]
    bounds.append((-float(w_test) * alpha, float(w_test) * (1.0 - alpha)))

    low = 0.0
    high = float(score_upper)
    threshold = float(w_test) * (1.0 - alpha)

    for _ in range(int(max_iter)):
        mid = (low + high) / 2.0
        c = np.append(-scores, -mid)
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
        if not res.success:
            # Return the conservative side if the LP fails.
            return float(high)
        eta_test = float(res.x[-1])
        if eta_test >= threshold - 1e-12:
            high = mid
        else:
            low = mid
        if high - low < 1e-7:
            break
    return float(high)


def lp_taus_by_unique_phi(
    data: ExperimentData,
    scores: np.ndarray,
    weights: np.ndarray,
    phi: np.ndarray,
    w_test: float,
    max_iter: int,
    n_jobs: int = 1,
) -> np.ndarray:
    """Compute thresholds for unique test membership patterns.

    The LP threshold depends only on the test point's group-membership vector,
    not on the individual image. For CIFAR-100 top-2 coarse groups this can mean
    up to about 190 unique patterns. The per-pattern LPs are independent, so this
    routine parallelizes them when --n_jobs > 1.
    """
    uniq, inv = unique_rows(data.test_phi)
    if len(uniq) == 0:
        return np.array([], dtype=float)

    phi_f = phi.astype(float)
    scores_f = scores.astype(float)
    weights_f = weights.astype(float)
    rows = [row.astype(float) for row in uniq]

    n_jobs_eff = int(n_jobs or 1)
    if n_jobs_eff <= 1 or len(rows) <= 1:
        tau_map = [
            solve_lp_threshold_for_phi(
                row, phi_f, scores_f, weights_f, float(w_test),
                float(data.alpha), float(data.score_upper), max_iter=max_iter,
            )
            for row in rows
        ]
    else:
        n_jobs_eff = min(n_jobs_eff, len(rows))
        with Pool(
            processes=n_jobs_eff,
            initializer=_lp_worker_init,
            initargs=(phi_f, scores_f, weights_f, float(w_test), float(data.alpha), float(data.score_upper), int(max_iter)),
        ) as pool:
            tau_map = pool.map(_lp_worker_solve, rows)

    tau_arr = np.asarray(tau_map, dtype=float)
    return tau_arr[inv]


def group_thresholds_exact(scores: np.ndarray, weights: np.ndarray, phi: np.ndarray, alpha: float) -> np.ndarray:
    d = phi.shape[1]
    taus = np.full(d, float("inf"), dtype=float)
    for g in range(d):
        idx = phi[:, g] > 0
        if np.any(idx):
            taus[g] = weighted_quantile(scores[idx], weights[idx], 1.0 - alpha)
    return taus


def compute_importance_weights_by_client(data: ExperimentData, bins: int = 20, clip: float = 50.0) -> List[np.ndarray]:
    """Importance weights adapted from label-shift/covariate-bin weighting.

    Classification: weights are based on empirical label ratios q(y)/p_k(y).
    Synthetic regression: weights are based on scalar covariate-bin ratios q(b)/p_k(b).
    The base federated weight pi_k/(n_k+1) is included.
    """
    n_k = np.array([len(s) for s in data.client_scores], dtype=int)
    out: List[np.ndarray] = []

    if data.client_labels is not None:
        labels_all = np.concatenate(data.client_labels) if data.n_cal > 0 else np.array([], dtype=int)
        n_classes = int(np.max(labels_all)) + 1 if labels_all.size else 1
        # Mixture target label distribution estimated from calibration using pi-weighted client histograms.
        q = np.zeros(n_classes, dtype=float)
        p_by_client = []
        for k, labels_k in enumerate(data.client_labels):
            counts = np.bincount(labels_k, minlength=n_classes).astype(float) + 1.0
            pk = counts / counts.sum()
            p_by_client.append(pk)
            q += data.pi[k] * pk
        q = q / q.sum()
        for k, labels_k in enumerate(data.client_labels):
            base = data.pi[k] / (n_k[k] + 1.0)
            if len(labels_k) == 0:
                out.append(np.array([], dtype=float))
                continue
            ratios = q[labels_k] / p_by_client[k][labels_k]
            ratios = np.clip(ratios, 1.0 / clip, clip)
            out.append(base * ratios.astype(float))
        return out

    # Synthetic/covariate fallback.
    x_all = np.concatenate(data.client_x_scalar) if data.n_cal > 0 else np.array([], dtype=float)
    if len(x_all) == 0:
        return [np.array([], dtype=float) for _ in range(data.K)]
    lo, hi = float(np.min(x_all)), float(np.max(x_all))
    edges = np.linspace(lo, hi + 1e-12, bins + 1)
    q = np.zeros(bins, dtype=float)
    p_by_client = []
    for k, xk in enumerate(data.client_x_scalar):
        counts, _ = np.histogram(xk, bins=edges)
        counts = counts.astype(float) + 1.0
        pk = counts / counts.sum()
        p_by_client.append(pk)
        q += data.pi[k] * pk
    q = q / q.sum()

    for k, xk in enumerate(data.client_x_scalar):
        base = data.pi[k] / (n_k[k] + 1.0)
        idx = np.digitize(xk, edges, right=False) - 1
        idx = np.clip(idx, 0, bins - 1)
        ratios = q[idx] / p_by_client[k][idx]
        ratios = np.clip(ratios, 1.0 / clip, clip)
        out.append(base * ratios.astype(float))
    return out


def run_methods(
    data: ExperimentData,
    seed: int,
    mc: int,
    method_keys: Sequence[str],
    deltas: Sequence[float],
    tdigest_K: int,
    group_rule: str,
    lp_max_iter: int,
    rng: np.random.Generator,
    n_jobs: int = 1,
) -> List[MethodRecord]:
    records: List[MethodRecord] = []

    flat_scores, flat_phi, flat_weights, flat_labels, client_ids = flatten_clients(data)
    n_k = np.array([len(s) for s in data.client_scores], dtype=int)
    weights_by_client = [np.full(n_k[k], data.pi[k] / (n_k[k] + 1.0), dtype=float) for k in range(data.K)]
    w_test = float(np.sum([data.pi[k] / (n_k[k] + 1.0) for k in range(data.K)]))
    active_atoms = int(len(set(row_tuples(flat_phi)))) if len(flat_phi) else 0
    base_complexity = (max(data.n_cal, 1) ** 1.5) * (max(data.n_groups, 1) ** 2)

    # CP
    if "cp" in method_keys:
        t0 = time.time()
        tau = split_cp_quantile(flat_scores, data.alpha)
        taus = np.full(data.n_test, tau, dtype=float)
        records.append(make_record(
            data, seed, mc, "cp", "Centralized CP", taus,
            atoms=0, coreset=data.n_cal, comm=data.n_cal, comp_speedup=float("nan"),
            runtime_sec=time.time() - t0,
            details={"modification": "standard split CP, marginal only"},
        ))

    # FedCP
    if "fedcp" in method_keys:
        t0 = time.time()
        tau = weighted_quantile_target(flat_scores, flat_weights, 1.0 - data.alpha)
        taus = np.full(data.n_test, tau, dtype=float)
        records.append(make_record(
            data, seed, mc, "fedcp", "FedCP", taus,
            atoms=0, coreset=data.n_cal, comm=data.n_cal, comp_speedup=float("nan"),
            runtime_sec=time.time() - t0,
            details={"modification": "weighted marginal FCP with lambda_k=pi_k/(n_k+1)"},
        ))

    # Mondrian CP + FedCP, group-level.
    if "mondrian" in method_keys:
        t0 = time.time()
        tau_global = weighted_quantile_target(flat_scores, flat_weights, 1.0 - data.alpha)
        group_taus = group_thresholds_exact(flat_scores, flat_weights, flat_phi, data.alpha)
        taus = combine_group_thresholds(data.test_phi, group_taus, tau_global, group_rule, rng)
        comm = int(np.sum(flat_phi))  # raw score may be sent to every group it belongs to.
        records.append(make_record(
            data, seed, mc, "mondrian", "Mondrian-FedCP", taus,
            atoms=0, coreset=int(np.sum(np.isfinite(group_taus))), comm=comm, comp_speedup=float("nan"),
            runtime_sec=time.time() - t0,
            details={
                "modification": "group-level FedCP thresholds on original groups, not atoms",
                "overlap_rule": group_rule,
            },
        ))

    # FedCF-style group-level baseline.
    if "fedcf" in method_keys:
        for delta in deltas[:1]:
            t0 = time.time()
            tau_global = weighted_quantile_target(flat_scores, flat_weights, 1.0 - data.alpha)
            group_taus, comm, coreset, impl = group_digest_thresholds_federated(
                data.client_scores, weights_by_client, data.client_phi, data.alpha, float(delta), tdigest_K
            )
            taus = combine_group_thresholds(data.test_phi, group_taus, tau_global, group_rule, rng)
            records.append(make_record(
                data, seed, mc, "fedcf", "FedCF-style", taus,
                atoms=0, coreset=coreset, comm=comm, comp_speedup=float("nan"),
                runtime_sec=time.time() - t0,
                details={
                    "modification": "FedCF adapted to original group-conditional evaluation via group digests",
                    "overlap_rule": group_rule,
                    "paper_delta": float(delta),
                    "digest_impl": impl,
                },
            ))

    # Importance-weighted FCP (Plassier-style adaptation).
    if "iw_fcp" in method_keys:
        t0 = time.time()
        iw_by_client = compute_importance_weights_by_client(data)
        iw_flat = np.concatenate(iw_by_client) if data.n_cal > 0 else np.array([], dtype=float)
        tau_global = weighted_quantile(flat_scores, iw_flat, 1.0 - data.alpha)
        group_taus = group_thresholds_exact(flat_scores, iw_flat, flat_phi, data.alpha)
        taus = combine_group_thresholds(data.test_phi, group_taus, tau_global, group_rule, rng)
        records.append(make_record(
            data, seed, mc, "iw_fcp", "IW-FCP", taus,
            atoms=0, coreset=data.n_cal, comm=int(np.sum(flat_phi)), comp_speedup=float("nan"),
            runtime_sec=time.time() - t0,
            details={
                "modification": "Plassier-style importance weighting adapted to original groups",
                "classification_weight": "estimated label-ratio q(y)/p_k(y)",
                "synthetic_weight": "estimated covariate-bin ratio q(b)/p_k(b)",
                "overlap_rule": group_rule,
            },
        ))

    # Centralized CondCP: uniform pooled weights.
    if "condcp" in method_keys:
        t0 = time.time()
        n = max(data.n_cal, 1)
        uni_weights = np.full(data.n_cal, 1.0 / (n + 1.0), dtype=float)
        uni_w_test = 1.0 / (n + 1.0)
        taus = lp_taus_by_unique_phi(data, flat_scores, uni_weights, flat_phi, uni_w_test, lp_max_iter, n_jobs=n_jobs)
        records.append(make_record(
            data, seed, mc, "condcp", "Centralized CondCP", taus,
            atoms=active_atoms, coreset=data.n_cal, comm=data.n_cal, comp_speedup=1.0,
            runtime_sec=time.time() - t0,
            details={"modification": "centralized CondCP with uniform pooled weights"},
        ))

    # Naive GC-FCP / Centralized GC-FCP: exact, no compression.
    if ("naive_gcfcp" in method_keys) or ("centralized_gcfcp" in method_keys):
        t0 = time.time()
        taus = lp_taus_by_unique_phi(data, flat_scores, flat_weights, flat_phi, w_test, lp_max_iter, n_jobs=n_jobs)
        display_name = "Naive GC-FCP" if "naive_gcfcp" in method_keys else "Centralized GC-FCP"
        records.append(make_record(
            data, seed, mc, "naive_gcfcp", display_name, taus,
            atoms=active_atoms, coreset=data.n_cal, comm=data.n_cal, comp_speedup=1.0,
            runtime_sec=time.time() - t0,
            details={
                "modification": "Centralized GC-FCP exact solver; no T-Digest compression",
                "same_as": "Naive GC-FCP algorithmically",
            },
        ))

    # Compressed GC-FCP.
    if "gcfcp" in method_keys:
        for delta in deltas:
            t0 = time.time()
            pseudo_s, pseudo_w, pseudo_phi, atoms, comm, impl = compress_by_atoms_federated(
                data.client_scores, weights_by_client, data.client_phi, float(delta), tdigest_K
            )
            if len(pseudo_s) == 0:
                taus = np.full(data.n_test, float("inf"), dtype=float)
                coreset = 0
                speedup = float("nan")
            else:
                temp = ExperimentData(
                    dataset=data.dataset,
                    score=data.score,
                    alpha=data.alpha,
                    group_names=data.group_names,
                    client_scores=data.client_scores,
                    client_phi=data.client_phi,
                    client_labels=data.client_labels,
                    client_x_scalar=data.client_x_scalar,
                    pi=data.pi,
                    test_scores_true=data.test_scores_true,
                    test_scores_all=data.test_scores_all,
                    test_phi=data.test_phi,
                    test_labels=data.test_labels,
                    set_kind=data.set_kind,
                    score_upper=data.score_upper,
                    metadata=data.metadata,
                )
                taus = lp_taus_by_unique_phi(temp, pseudo_s, pseudo_w, pseudo_phi, w_test, lp_max_iter, n_jobs=n_jobs)
                coreset = int(len(pseudo_s))
                speedup = float((max(data.n_cal, 1) / max(coreset, 1)) ** 1.5)
            records.append(make_record(
                data, seed, mc, f"gcfcp_delta_{fmt_delta(float(delta))}", f"GC-FCP (delta={fmt_delta(float(delta))})", taus,
                atoms=atoms, coreset=coreset, comm=comm, comp_speedup=speedup,
                runtime_sec=time.time() - t0,
                details={
                    "paper_delta": float(delta),
                    "tdigest_K": int(tdigest_K),
                    "tdigest_package_delta": float(tdigest_K) / float(delta),
                    "digest_impl": impl,
                },
            ))

    return records


# -------------------------
# Table generation
# -------------------------

METRIC_COLUMNS = [
    "marg_cov", "worst_group_cov", "avg_group_cov", "max_group_gap",
    "avg_set_size", "norm_avg_set_size", "atoms", "coreset", "comm", "comp_speedup"
]


def mean_se(series: pd.Series) -> Tuple[float, float]:
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(arr))
    se = float(np.std(arr, ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return mean, se


def fmt_mean_se(mean: float, se: float, digits: int = 3, integer: bool = False) -> str:
    if not np.isfinite(mean):
        return "N/A"
    if integer:
        return f"{mean:.1f}"
    if se is None or not np.isfinite(se) or se == 0:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} +/- {se:.{digits}f}"



def _parse_json_array(value: Any) -> np.ndarray:
    """Parse a JSON-encoded numeric vector from raw_results.csv."""
    if isinstance(value, str):
        try:
            return np.asarray(json.loads(value), dtype=float)
        except Exception:
            return np.array([], dtype=float)
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=float)
    return np.array([], dtype=float)


def pooled_group_summary(sub: pd.DataFrame, alpha: float) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """Pool group coverage numerators/denominators across MC repetitions.

    This reports min_G sum_m covered_m(G) / sum_m n_m(G), rather than
    mean_m min_G cov_m(G), which is downward biased for many groups with
    moderate per-run denominators.
    """
    numer: Optional[np.ndarray] = None
    denom: Optional[np.ndarray] = None
    for _, row in sub.iterrows():
        cov = _parse_json_array(row.get("group_cov_json", "[]"))
        size = _parse_json_array(row.get("group_size_json", "[]"))
        if cov.size == 0 or size.size == 0:
            continue
        m = min(cov.size, size.size)
        cov = cov[:m]
        size = size[:m]
        valid = np.isfinite(cov) & np.isfinite(size) & (size > 0)
        if not np.any(valid):
            continue
        if numer is None:
            numer = np.zeros(m, dtype=float)
            denom = np.zeros(m, dtype=float)
        elif len(numer) < m:
            numer = np.pad(numer, (0, m - len(numer)), constant_values=0.0)
            denom = np.pad(denom, (0, m - len(denom)), constant_values=0.0)
        numer[:m][valid] += cov[valid] * size[valid]
        denom[:m][valid] += size[valid]

    if numer is None or denom is None or not np.any(denom > 0):
        return float("nan"), float("nan"), float("nan"), np.array([], dtype=float), np.array([], dtype=float)
    pooled = np.full_like(numer, np.nan, dtype=float)
    valid = denom > 0
    pooled[valid] = numer[valid] / denom[valid]
    worst = float(np.nanmin(pooled[valid]))
    avg = float(np.average(pooled[valid], weights=denom[valid]))
    max_gap = float(np.nanmax(np.abs(pooled[valid] - (1.0 - float(alpha)))))
    return worst, avg, max_gap, pooled, denom

def aggregate_table(df: pd.DataFrame, score_filter: Optional[Iterable[str]] = None, method_keys: Optional[Iterable[str]] = None) -> pd.DataFrame:
    work = df.copy()
    if score_filter is not None:
        sf = set(score_filter)
        work = work[work["score"].isin(sf)]
    if method_keys is not None:
        mk = set(method_keys)
        work = work[work["method_key"].isin(mk)]

    if work.empty:
        return pd.DataFrame(columns=[
            "Score", "Method", "Marg. cov.", "Worst-group cov.", "Avg group cov.",
            "Max group gap", "Avg set size", "Norm set size", "Atoms", "Coreset", "Comm.", "Comp. Speedup"
        ])

    rows: List[Dict[str, str]] = []
    group_cols = ["score", "method_key", "method"]
    for (score, method_key, method), sub in work.groupby(group_cols, sort=False):
        row: Dict[str, str] = {"Score": str(score).upper(), "Method": str(method)}

        m, se = mean_se(sub["marg_cov"])
        row["Marg. cov."] = fmt_mean_se(m, se, digits=3)

        alpha = 0.1
        if "alpha" in sub:
            alpha_vals = pd.to_numeric(sub["alpha"], errors="coerce").dropna()
            if len(alpha_vals) > 0:
                alpha = float(alpha_vals.iloc[0])
        pooled_worst, pooled_avg, pooled_max_gap, _, _ = pooled_group_summary(sub, alpha)
        row["Worst-group cov."] = fmt_mean_se(pooled_worst, 0.0, digits=3)
        row["Avg group cov."] = fmt_mean_se(pooled_avg, 0.0, digits=3)
        row["Max group gap"] = fmt_mean_se(pooled_max_gap, 0.0, digits=3)

        m, se = mean_se(sub["avg_set_size"])
        row["Avg set size"] = fmt_mean_se(m, se, digits=3)

        if "norm_avg_set_size" in sub.columns:
            m, se = mean_se(sub["norm_avg_set_size"])
            row["Norm set size"] = fmt_mean_se(m, se, digits=4)
        else:
            row["Norm set size"] = "N/A"

        for metric, label in [
            ("atoms", "Atoms"),
            ("coreset", "Coreset"),
            ("comm", "Comm."),
        ]:
            m, se = mean_se(sub[metric])
            row[label] = fmt_mean_se(m, se, digits=1, integer=True)

        m, se = mean_se(sub["comp_speedup"])
        if np.isfinite(m):
            row["Comp. Speedup"] = fmt_mean_se(m, se, digits=2) + "x"
        else:
            row["Comp. Speedup"] = "N/A"

        rows.append(row)

    cols = ["Score", "Method", "Marg. cov.", "Worst-group cov.", "Avg group cov.",
            "Max group gap", "Avg set size", "Norm set size", "Atoms", "Coreset", "Comm.", "Comp. Speedup"]
    return pd.DataFrame(rows)[cols]

def select_gcfcp_keys(df: pd.DataFrame) -> List[str]:
    return [k for k in df["method_key"].drop_duplicates().tolist() if str(k).startswith("gcfcp_delta_")]


def load_results(results_csv: Path | str) -> pd.DataFrame:
    """Read a saved raw_results.csv file."""
    return pd.read_csv(Path(results_csv))


def make_main_thr_raps_table(df: pd.DataFrame) -> pd.DataFrame:
    """Main zPQL table with THR and RAPS/RAPR."""
    gcfcp_keys = select_gcfcp_keys(df)
    main_methods = ["cp", "fedcp", "mondrian", "fedcf", "iw_fcp", "naive_gcfcp"] + gcfcp_keys
    return aggregate_table(df, score_filter=["thr", "raps", "rapr"], method_keys=main_methods)


def make_aps_table(df: pd.DataFrame) -> pd.DataFrame:
    """Small zPQL table with APS."""
    gcfcp_keys = select_gcfcp_keys(df)
    first_gcfcp = gcfcp_keys[:3]
    aps_methods = ["cp", "fedcp", "fedcf", "iw_fcp", "naive_gcfcp"] + first_gcfcp
    return aggregate_table(df, score_filter=["aps"], method_keys=aps_methods)

def make_ablation_group_conditioning_table(df: pd.DataFrame) -> pd.DataFrame:
    """HC6E table isolating group conditioning."""
    gcfcp_keys = select_gcfcp_keys(df)
    first_gcfcp = gcfcp_keys[:1]
    group_methods = ["fedcp", "mondrian", "fedcf"] + first_gcfcp
    return aggregate_table(df, score_filter=["thr", "reg"], method_keys=group_methods)


def make_ablation_compression_table(df: pd.DataFrame) -> pd.DataFrame:
    """HC6E table isolating compression."""
    gcfcp_keys = select_gcfcp_keys(df)
    compression_methods = ["naive_gcfcp"] + gcfcp_keys
    return aggregate_table(df, score_filter=["thr", "reg"], method_keys=compression_methods)


def make_ablation_federated_aggregation_table(df: pd.DataFrame) -> pd.DataFrame:
    """HC6E table isolating federated aggregation."""
    gcfcp_keys = select_gcfcp_keys(df)
    first_gcfcp = gcfcp_keys[:1]
    federated_methods = ["condcp", "naive_gcfcp"] + first_gcfcp
    return aggregate_table(df, score_filter=["thr", "reg"], method_keys=federated_methods)


def make_all_tables(results_csv: Path, out_dir: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    df = load_results(results_csv)
    if out_dir is None:
        out_dir = results_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    tables: Dict[str, pd.DataFrame] = {}
    tables["main_thr_raps"] = make_main_thr_raps_table(df)
    tables["aps_small"] = make_aps_table(df)
    tables["ablation_group_conditioning"] = make_ablation_group_conditioning_table(df)
    tables["ablation_compression"] = make_ablation_compression_table(df)
    tables["ablation_federated_aggregation"] = make_ablation_federated_aggregation_table(df)

    for name, tab in tables.items():
        csv_path = out_dir / f"{name}.csv"
        md_path = out_dir / f"{name}.md"
        tab.to_csv(csv_path, index=False)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(tab.to_markdown(index=False))
            f.write("\n")
    return tables


# -------------------------
# Command runner
# -------------------------

def default_methods_for_args(args: argparse.Namespace) -> List[str]:
    if args.methods:
        return [m.strip() for m in args.methods.split(",") if m.strip()]
    methods = ["cp", "fedcp", "mondrian", "fedcf", "iw_fcp", "gcfcp"]
    if args.include_exact:
        methods.extend(["condcp", "naive_gcfcp"])
    return methods


def default_scores_for_dataset(dataset: str, scores_arg: str) -> List[str]:
    if scores_arg:
        return [s.strip().lower() for s in scores_arg.split(",") if s.strip()]
    if dataset == "synthetic":
        return ["reg"]
    return ["thr", "raps", "aps"]


def run_experiment(args: argparse.Namespace) -> Path:
    out_dir = Path(args.out_dir)
    stamp = time.strftime("%Y%m%d_%H")
    run_dir = out_dir / f"rebuttal_{args.dataset}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    K = int(args.clients)
    pi = normalize_pi(parse_csv_list(args.pi, float) if args.pi else None, K)
    scores = default_scores_for_dataset(args.dataset, args.scores)
    deltas = parse_csv_list(args.deltas, float)
    if not deltas:
        deltas = [100.0]
    method_keys = default_methods_for_args(args)

    all_records: List[MethodRecord] = []
    config = vars(args).copy()
    config["pi"] = pi.tolist()
    config["scores_resolved"] = scores
    config["methods_resolved"] = method_keys
    config["deltas_resolved"] = deltas
    config["notes"] = {
        "delta_convention": "CLI deltas are manuscript deltas; TDigest package delta=tdigest_K/delta.",
        "overlap_group_rule": args.group_rule,
        "overlap_group_rule_meaning": "For group-level baselines, min/max refer to active group indices, not threshold values.",
        "FedCF_style": "group-level adaptation evaluated on original groups, not atoms.",
        "pooled_worst_group_coverage": "Tables compute worst-group coverage after pooling group numerators/denominators across MC repetitions.",
        "raw_group_set_size_columns": "raw_results.csv includes group_avg_set_size_json and group_norm_avg_set_size_json, aligned with group_cov_json/group_size_json.",
        "IW_FCP": "importance-weighted baseline adapted to original groups; classification uses label-ratio weights.",
    }

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    for mc in range(int(args.times)):
        seed = int(args.base_seed) + mc
        set_all_seeds(seed)
        rng = np.random.default_rng(seed + 999)

        for score in scores:
            score_eff = "reg" if args.dataset == "synthetic" else score.lower()
            if args.dataset == "synthetic":
                data = make_synthetic_data(
                    seed=seed,
                    alpha=float(args.alpha),
                    K=K,
                    pi=pi,
                    n_cal=int(args.n_cal),
                    n_test=int(args.n_test),
                    synthetic_train=int(args.synthetic_train),
                    poly_degree=int(args.poly_degree),
                    label_shift=not args.no_synthetic_label_shift,
                )
            else:
                data = make_real_data(
                    seed=seed,
                    dataset=args.dataset,
                    score=score_eff,
                    alpha=float(args.alpha),
                    K=K,
                    pi=pi,
                    n_cal=int(args.n_cal),
                    n_test=int(args.n_test),
                    group_mode=args.group_mode,
                    top_r=int(args.top_r),
                    raps_lambda=float(args.raps_lambda),
                    raps_k_reg=int(args.raps_k_reg),
                    batch_size=int(args.batch_size),
                    x_feature=str(args.x_feature),
                    client_split=str(args.client_split),
                    dirichlet_beta=float(args.dirichlet_beta),
                    min_client_count=int(args.min_client_count),
                    data_root=str(args.data_root),
                    imagenet_val_dir=str(args.imagenet_val_dir) if args.imagenet_val_dir else "",
                    imagenet_class_index_json=str(args.imagenet_class_index_json) if args.imagenet_class_index_json else "",
                    imagenet_meta_map=str(args.imagenet_meta_map) if args.imagenet_meta_map else "",
                    imagenet_model=str(args.imagenet_model),
                    imagenet_cache=str(args.imagenet_cache) if getattr(args, "imagenet_cache", "") else "",
                    ambiguity_score=str(args.ambiguity_score),
                    ambiguity_bins=int(args.ambiguity_bins),
                    ambiguity_binning=str(args.ambiguity_binning),
                )

            recs = run_methods(
                data=data,
                seed=seed,
                mc=mc,
                method_keys=method_keys,
                deltas=deltas,
                tdigest_K=int(args.tdigest_K),
                group_rule=args.group_rule,
                lp_max_iter=int(args.lp_max_iter),
                rng=rng,
                n_jobs=int(args.n_jobs),
            )
            all_records.extend(recs)

            if not args.quiet:
                print(f"[mc={mc+1}/{args.times}] dataset={args.dataset} score={score_eff} records={len(recs)}")

            # Save incremental results so long jobs are recoverable.
            tmp_df = pd.DataFrame([asdict(r) for r in all_records])
            tmp_df.to_csv(run_dir / "raw_results.csv", index=False)

    results_csv = run_dir / "raw_results.csv"
    df = pd.DataFrame([asdict(r) for r in all_records])
    df.to_csv(results_csv, index=False)

    tables = make_all_tables(results_csv, run_dir)
    if not args.quiet:
        for name, tab in tables.items():
            print("\n" + "=" * 100)
            print(name)
            print("=" * 100)
            print(tab.to_markdown(index=False))

    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GC-FCP rebuttal experiments.")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="Run experiments and save raw data + tables.")
    run.add_argument("--dataset", type=str, default="synthetic", choices=["synthetic", "cifar10", "cifar100", "imagenet"])
    run.add_argument("--scores", type=str, default="", help="Comma-separated scores. Real-data default: thr,raps,aps. Main table uses THR+RAPS; small table uses APS.")
    run.add_argument("--clients", type=int, default=50, help="Number of clients. Use 50 for CIFAR-100 rebuttal.")
    run.add_argument("--times", type=int, default=20, help="Monte Carlo repetitions.")
    run.add_argument("--alpha", type=float, default=0.1)
    run.add_argument("--n_cal", type=int, default=5000)
    run.add_argument("--n_test", type=int, default=5000)
    run.add_argument("--pi", type=str, default="", help="Comma-separated mixture weights. Defaults to uniform.")
    run.add_argument("--base_seed", type=int, default=114514)
    run.add_argument("--out_dir", type=str, default="results_rebuttal")
    run.add_argument("--quiet", action="store_true")
    run.add_argument("--n_jobs", type=int, default=max(1, min(8, cpu_count())),
                     help="Parallel worker processes for LP solves over unique test group patterns. Use 1 to disable.")

    # Methods and delta.
    run.add_argument("--methods", type=str, default="", help="Comma-separated method keys. Available: cp,fedcp,mondrian,fedcf,iw_fcp,condcp,naive_gcfcp,centralized_gcfcp,gcfcp. Empty uses default.")
    run.add_argument("--include_exact", action="store_true", help="Include Centralized CondCP and Naive GC-FCP exact LP baselines.")
    run.add_argument("--deltas", type=str, default="100", help="Comma-separated manuscript T-Digest deltas, e.g. 50,100,250.")
    run.add_argument("--tdigest_K", type=int, default=25, help="Internal TDigest K parameter.")
    run.add_argument("--group_rule", type=str, default="min", choices=["min", "max", "random"], help="Overlap assignment rule for group-level baselines. min chooses the smallest active group index; max chooses the largest active group index; random samples one active group.")
    run.add_argument("--lp_max_iter", type=int, default=35)

    # Synthetic.
    run.add_argument("--synthetic_train", type=int, default=1000)
    run.add_argument("--poly_degree", type=int, default=4)
    run.add_argument("--no_synthetic_label_shift", action="store_true")

    # Real data.
    run.add_argument("--group_mode", type=str, default="coarse_topr",
                     choices=["coarse_topr", "imagenet_meta_topr", "imagenet_semantic_ambiguity", "imagenet_semantic_margin", "pred_window", "disjoint_pred"],
                     help="CIFAR-100 uses semantic coarse_topr by default. ImageNet additionally supports imagenet_semantic_ambiguity: 10 top-1 semantic meta-groups plus ambiguity bins. CIFAR-10 uses --x_feature groups from realdata_additional.py; set --x_feature pred_label for the paper-style predicted-label windows.")
    run.add_argument("--x_feature", type=str, default="confidence", choices=["pred_label", "confidence", "entropy"],
                     help="Scalar feature used for CIFAR-10 group construction. Default confidence follows the realdata_additional.py difficulty-style grouping; pred_label recovers the paper-style predicted-label windows.")
    run.add_argument("--client_split", type=str, default="auto", choices=["auto", "label", "dirichlet"],
                     help="Client allocation for real data. auto keeps the original label split except CIFAR-10 with K > 10, where it switches to Dirichlet sample-level non-i.i.d. allocation to avoid empty clients.")
    run.add_argument("--dirichlet_beta", type=float, default=0.3,
                     help="Dirichlet concentration for --client_split dirichlet. Smaller values create stronger label skew.")
    run.add_argument("--min_client_count", type=int, default=5,
                     help="Minimum calibration samples per client for Dirichlet allocation.")
    run.add_argument("--top_r", type=int, default=2, help="Top-r coarse/meta groups for CIFAR-100 and ImageNet meta-top-r.")
    run.add_argument("--ambiguity_score", type=str, default="margin", choices=["margin", "entropy", "confidence"],
                     help="Scalar ambiguity score for --group_mode imagenet_semantic_ambiguity. margin uses top-1 minus top-2 probability; entropy uses predictive entropy; confidence uses max probability.")
    run.add_argument("--ambiguity_bins", type=int, default=5,
                     help="Number of ambiguity/difficulty bins for --group_mode imagenet_semantic_ambiguity.")
    run.add_argument("--ambiguity_binning", type=str, default="quantile", choices=["quantile", "uniform"],
                     help="How to construct ambiguity bins. quantile uses the calibration split; uniform uses the natural score range.")
    run.add_argument("--batch_size", type=int, default=128)
    run.add_argument("--data_root", type=str, default="./data", help="Root for CIFAR data, or ImageNet root when --dataset imagenet.")
    run.add_argument("--imagenet_val_dir", type=str, default="", help="Optional ImageNet validation ImageFolder directory. If omitted, tries torchvision.datasets.ImageNet(root=--data_root, split=val).")
    run.add_argument("--imagenet_class_index_json", type=str, default="", help="Optional ImageNet class-index JSON for ImageFolder fallback, e.g. {idx:[wnid,name]} or {folder_name: idx}.")
    run.add_argument("--imagenet_meta_map", type=str, default="", help="Optional mapping from ImageNet class index/category/wnid to one of 10 meta-groups. JSON list or dict.")
    run.add_argument("--imagenet_model", type=str, default="resnet50", choices=["resnet50"], help="TorchVision ImageNet-1K pretrained model.")
    run.add_argument("--imagenet_cache", type=str, default="", help="Optional NPZ cache made by the cache_imagenet command. When provided, ImageNet runs use cached labels/probabilities and skip image loading/model inference.")
    run.add_argument("--raps_lambda", type=float, default=0.01)
    run.add_argument("--raps_k_reg", type=int, default=5)


    cache = sub.add_parser("cache_imagenet", help="Precompute ImageNet validation labels/probabilities for fast repeated runs.")
    cache.add_argument("--data_root", type=str, default="./data", help="ImageNet root for torchvision.datasets.ImageNet, or parent containing val/.")
    cache.add_argument("--imagenet_val_dir", type=str, default="", help="Optional ImageFolder-style validation directory. If omitted, tries torchvision.datasets.ImageNet(root=--data_root, split=val), then --data_root/val.")
    cache.add_argument("--imagenet_class_index_json", type=str, default="", help="Optional ImageNet class-index JSON for ImageFolder fallback.")
    cache.add_argument("--imagenet_model", type=str, default="resnet50", choices=["resnet50"], help="TorchVision ImageNet-1K pretrained model.")
    cache.add_argument("--batch_size", type=int, default=256)
    cache.add_argument("--num_workers", type=int, default=max(0, min(8, cpu_count())))
    cache.add_argument("--pin_memory", action="store_true", help="Use pinned DataLoader memory when running on CUDA.")
    cache.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    cache.add_argument("--cache_path", type=str, required=True, help="Output NPZ path, e.g. data/imagenet_resnet50_val_probs.npz.")
    cache.add_argument("--compressed", action="store_true", help="Use np.savez_compressed. Smaller file, slower to write/read.")
    cache.add_argument("--log_every", type=int, default=20)
    cache.add_argument("--quiet", action="store_true")

    tables = sub.add_parser("tables", help="Regenerate tables from a raw_results.csv file.")
    tables.add_argument("--results_csv", type=str, required=True)
    tables.add_argument("--out_dir", type=str, default="")

    smoke = sub.add_parser("smoke", help="Run a small synthetic correctness check.")
    smoke.add_argument("--out_dir", type=str, default="results_rebuttal_smoke")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        args.command = "run"

    if args.command == "run":
        run_dir = run_experiment(args)
        print(f"Saved run directory: {run_dir}")
    elif args.command == "cache_imagenet":
        cache_path = cache_imagenet_probabilities(args)
        print(f"Saved ImageNet cache: {cache_path}")
    elif args.command == "tables":
        results_csv = Path(args.results_csv)
        out_dir = Path(args.out_dir) if args.out_dir else results_csv.parent
        tables = make_all_tables(results_csv, out_dir)
        for name, tab in tables.items():
            print("\n" + "=" * 100)
            print(name)
            print("=" * 100)
            print(tab.to_markdown(index=False))
    elif args.command == "smoke":
        smoke_args = argparse.Namespace(
            command="run",
            dataset="synthetic",
            scores="reg",
            clients=4,
            times=2,
            alpha=0.1,
            n_cal=120,
            n_test=60,
            pi="",
            base_seed=123,
            out_dir=args.out_dir,
            quiet=False,
            n_jobs=2,
            methods="cp,fedcp,mondrian,fedcf,iw_fcp,condcp,naive_gcfcp,gcfcp",
            include_exact=True,
            deltas="20,50",
            tdigest_K=25,
            group_rule="min",
            lp_max_iter=25,
            synthetic_train=300,
            poly_degree=4,
            no_synthetic_label_shift=False,
            group_mode="coarse_topr",
            x_feature="confidence",
            client_split="auto",
            dirichlet_beta=0.3,
            min_client_count=5,
            top_r=2,
            batch_size=128,
            raps_lambda=0.01,
            raps_k_reg=5,
        )
        run_dir = run_experiment(smoke_args)
        print(f"Smoke test saved run directory: {run_dir}")
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()

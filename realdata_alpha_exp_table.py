import argparse
import json
import os
import random
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from tdigest import TDigest

import torch
import torch.hub
import torchvision
import torchvision.transforms as transforms

# --------------------
# Environment / global
# --------------------
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
torch.hub.set_dir('./model')
os.makedirs('results', exist_ok=True)


class TableExperiment:
    """Real-data benchmarks for GC-FCP and baselines (CIFAR10 / PathMNIST).

    The experiment fixes alpha=0.1, runs Monte Carlo repetitions (random split + bootstrap),
    and outputs a table reporting marginal and group-conditional coverage / set size.

    Methods reported:
      - Centralized CP
      - FedCP
      - Centralized CondCP
      - Centralized GC-FCP
      - GC-FCP (δ in a user-specified list)

    Acceleration is computed relative to Centralized CondCP using per-test-point wall time
    for methods that require per-test LP solves; Centralized CP and FedCP are excluded.
    """

    ALPHA: float = 0.1

    def __init__(
        self,
        dataset: str = 'cifar10',
        overlap: bool = True,
        K: int = 5,
        pi_k: Optional[List[float]] = None,
        num_mc: int = 10,
        n_jobs: int = 10,
        gcfcp_deltas: Optional[List[float]] = None,
        tdigest_K: int = 25,
        model_path: Optional[str] = None,
        base_seed: int = 114514,
    ):
        self.dataset = (dataset or 'cifar10').lower()
        self.overlap = bool(overlap)
        self.K = int(K)
        self.pi_k = pi_k if pi_k is not None else [1.0 / self.K] * self.K
        self.num_mc = int(num_mc)
        self.n_jobs = cpu_count() if n_jobs == -1 else int(n_jobs)
        self.base_seed = int(base_seed)

        # δ values used for GC-FCP rows in the table
        self.gcfcp_deltas = gcfcp_deltas if gcfcp_deltas is not None else [1, 0.1, 0.01]
        self.tdigest_K = int(tdigest_K)
        self.model_path = model_path

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # ---------- dataset/model-specific initialization ----------
        if self.dataset == 'cifar10':
            self.model = torch.hub.load(
                "chenyaofo/pytorch-cifar-models",
                "cifar10_resnet56",
                pretrained=True,
            )
            self.model.to(self.device).eval()
            self.normalize = transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761],
            )
            self.n_classes = 10

            if self.overlap:
                # 4 overlapping groups (matches the user-provided table layout)
                self.groups = [[0, 4], [2, 6], [4, 8], [6, 10]]
                self.eval_ranges = ['G1', 'G2', 'G3', 'G4']
            else:
                # disjoint fallback
                self.groups = [[0, 2], [2, 4], [4, 6], [6, 8], [8, 10]]
                self.eval_ranges = ['G1', 'G2', 'G3', 'G4', 'G5']

            # client label partitions: equal split of 10 classes
            per = int(self.n_classes / self.K)
            all_labels = list(range(self.n_classes))
            self.client_label_sets = [set(all_labels[i * per:(i + 1) * per]) for i in range(self.K)]

        elif self.dataset == 'pathmnist':
            # Lazy imports for CIFAR-only environments
            from medmnist import INFO  # type: ignore
            import medmnist  # type: ignore
            try:
                from medmnist_train import SimpleCNN  # type: ignore
            except Exception as e:
                raise ImportError("SimpleCNN not found. Ensure medmnist_train.py is available.") from e

            self.model = SimpleCNN(n_classes=9).to(self.device)
            if self.model_path is not None:
                ckpt = torch.load(self.model_path, map_location=self.device)
                if isinstance(ckpt, dict) and 'model_state' in ckpt:
                    self.model.load_state_dict(ckpt['model_state'])
                elif isinstance(ckpt, dict):
                    self.model.load_state_dict(ckpt)
                else:
                    self.model = ckpt.to(self.device)
            self.model.eval()

            self.normalize = lambda x: x  # Identity for 0-1 tensors
            self.n_classes = 9

            if self.overlap:
                # 5 overlapping groups as in the demo
                self.groups = [[0, 3], [1, 4], [2, 5], [3, 6], [4, 9]]
                self.eval_ranges = ['G1', 'G2', 'G3', 'G4', 'G5']
            else:
                self.groups = [[0, 2], [2, 4], [4, 6], [6, 8], [8, 9]]
                self.eval_ranges = ['G1', 'G2', 'G3', 'G4', 'G5']

            # client label holdings
            self.client_label_sets = [{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8}]

        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")

        # method list for the table
        self.methods: List[str] = (
            # ['Centralized CP', 'FedCP', 'Centralized CondCP', 'Centralized GC-FCP']
            ['Centralized CP', 'FedCP']
            + [f'GC-FCP (δ={self._fmt_delta(d)})' for d in self.gcfcp_deltas]
        )

        # results populated by run()
        self.avg_miscov: Dict[str, np.ndarray] = {}
        self.avg_set_size: Dict[str, np.ndarray] = {}
        self.avg_time_per_test: Dict[str, Optional[float]] = {}

        # Standard errors over Monte Carlo repetitions. These are used in the
        # reviewer-facing table so entries can be reported as mean +- stderr.
        self.avg_miscov_stderr: Dict[str, np.ndarray] = {}
        self.avg_set_size_stderr: Dict[str, np.ndarray] = {}
        self.avg_time_per_test_stderr: Dict[str, Optional[float]] = {}

        # Raw numerical outputs retained after run(), with shape
        #   raw_miscov[method]   : (num_mc, 1 + num_groups)
        #   raw_set_size[method] : (num_mc, 1 + num_groups)
        #   raw_time_per_test[method] : (num_mc,)
        # The first column is marginal; the remaining columns are G1, G2, ... .
        self.raw_miscov: Dict[str, np.ndarray] = {}
        self.raw_set_size: Dict[str, np.ndarray] = {}
        self.raw_time_per_test: Dict[str, np.ndarray] = {}
        self.run_seeds: List[int] = []

    @staticmethod
    def _fmt_delta(delta: float) -> str:
        return str(int(delta)) if float(delta).is_integer() else f"{delta:g}"

    @staticmethod
    def _stderr(values: np.ndarray, axis: int = 0) -> np.ndarray:
        """Nan-aware standard error over Monte Carlo repetitions.

        If only one finite value is available, the standard error is set to 0.
        This keeps smoke-test runs with --times 1 printable while still giving
        the usual sample standard error for reviewer-facing multi-run tables.
        """
        arr = np.asarray(values, dtype=float)
        counts = np.sum(np.isfinite(arr), axis=axis)
        std = np.nanstd(arr, axis=axis, ddof=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            se = std / np.sqrt(counts)
        return np.where(counts > 1, se, 0.0)

    @staticmethod
    def _fmt_mean_se(mean: float, se: float, digits: int = 3) -> str:
        if mean is None or not np.isfinite(mean):
            return 'N/A'
        if se is None or not np.isfinite(se):
            return f"{mean:.{digits}f}"
        return f"{mean:.{digits}f} +- {se:.{digits}f}"

    @staticmethod
    def _safe_key(text: str) -> str:
        return ''.join(ch if ch.isalnum() else '_' for ch in text).strip('_')

    def _metric_specs(self) -> List[Tuple[int, str, str, Optional[float], Optional[float]]]:
        """Return metric columns: index, scope, label, lower bound, upper bound."""
        specs: List[Tuple[int, str, str, Optional[float], Optional[float]]] = [
            (0, 'marginal', 'Marginal', None, None)
        ]
        for gi, (low_b, up_b) in enumerate(self.groups):
            specs.append((gi + 1, 'group', f'G{gi + 1}', float(low_b), float(up_b)))
        return specs

    def _set_all_seeds(self, seed: int) -> None:
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

    def _load_pathmnist_items(self) -> List[Tuple[torch.Tensor, int]]:
        from medmnist import INFO  # type: ignore
        import medmnist  # type: ignore

        transform = transforms.Compose([transforms.ToTensor()])
        data_flag = 'pathmnist'
        info = INFO[data_flag]
        DataClass = getattr(medmnist, info['python_class'])
        val_ds = DataClass(split='val', transform=transform, download=True, root='data')
        test_ds = DataClass(split='test', transform=transform, download=True, root='data')

        def _extract(ds):
            out = []
            for i in range(len(ds)):
                x, y = ds[i]
                if isinstance(y, np.ndarray):
                    y_int = int(y) if y.ndim == 0 else (int(y.item()) if (y.ndim == 1 and y.size == 1) else int(np.argmax(y)))
                elif torch.is_tensor(y):
                    y_int = int(y.item()) if y.ndim <= 1 else int(torch.argmax(y).item())
                else:
                    y_int = int(y)
                out.append((x, y_int))
            return out

        return _extract(val_ds) + _extract(test_ds)

    def compute_scores_and_preds(
        self,
        images: List[torch.Tensor],
        labels: List[int],
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """Return (scores, predicted-class, class-probabilities).

        Nonconformity score is S = 1 - P_
        """
        scores: List[float] = []
        preds: List[int] = []
        probs: List[np.ndarray] = []

        batch_size = 128
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch_x = torch.stack(images[i:i + batch_size]).to(self.device)
                batch_x = self.normalize(batch_x)
                batch_y = torch.tensor(labels[i:i + batch_size]).to(self.device)

                logits = self.model(batch_x)
                soft = torch.softmax(logits, dim=1)
                prob_y = soft.gather(1, batch_y.unsqueeze(1)).squeeze(1).cpu().numpy()

                scores.extend((1.0 - prob_y).tolist())
                preds.extend(logits.argmax(1).cpu().numpy().tolist())
                probs.extend(soft.cpu().numpy())

        return np.array(scores, dtype=float), np.array(preds, dtype=float), probs

    def compute_vanilla_quantile(self, S_calib: List[np.ndarray]) -> float:
        """Centralized CP: global quantile with (1+1/n) correction."""
        S = np.concatenate(S_calib)
        n = len(S)
        q = np.quantile(S, (1 - self.ALPHA) * (1 + 1 / n))
        return float(min(max(q, 0.0), 1.0))

    def compute_fedcp_quantile(self, S_calib: List[np.ndarray], lambda_k: List[float]) -> float:
        """FedCP: weighted quantile of pooled scores using client weights lambda_k."""
        S_all = np.concatenate(S_calib)
        w_all = np.concatenate([np.full(len(S_calib[k]), lambda_k[k], dtype=float) for k in range(self.K)])

        order = np.argsort(S_all)
        S_s = S_all[order]
        w_s = w_all[order]
        cum = np.cumsum(w_s)
        target = 1.0 - self.ALPHA

        if cum[-1] < target:
            return 1.0
        idx = int(np.searchsorted(cum, target, side='left'))
        q = float(S_s[min(idx, len(S_s) - 1)])
        return float(min(max(q, 0.0), 1.0))

    # -----------------
    # LP-based thresholds
    # -----------------
    @staticmethod
    def find_S_star(
        X_test_point: float,
        X_all: np.ndarray,
        S_all: np.ndarray,
        w_all: np.ndarray,
        w_test: float,
        alpha: float,
        groups: List[List[int]],
    ) -> float:
        """Solve the centralized dual LP by bisection on the score threshold."""
        d = len(groups)
        Phi_test = np.array([1 if g[0] <= X_test_point < g[1] else 0 for g in groups])
        Phi_all = np.array([[1 if g[0] <= X_all[i] < g[1] else 0 for g in groups] for i in range(len(X_all))])

        A_eq = np.hstack((Phi_all.T, Phi_test.reshape(-1, 1)))
        b_eq = np.zeros(d)
        bounds = [(-w_all[i] * alpha, w_all[i] * (1 - alpha)) for i in range(len(X_all))]
        bounds.append((-w_test * alpha, w_test * (1 - alpha)))

        low, high = 0.0, 1.0
        for _ in range(50):
            mid = (low + high) / 2
            c = np.append(-S_all, -mid)
            res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            if not res.success:
                return float(low)
            eta_test_val = res.x[-1]
            if eta_test_val >= w_test * (1 - alpha):
                high = mid
            else:
                low = mid
            if high - low < 1e-8:
                break
        return float(high)

    def compute_gcfcp_pseudo_data_fed(
        self,
        X_calib: List[np.ndarray],
        S_calib: List[np.ndarray],
        w_calib: List[np.ndarray],
        delta: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Construct GC-FCP pseudo-data (scores, weights, group-features) via federated TDigest."""
        breakpoints = sorted(set(sum(self.groups, [])))
        atoms = [(breakpoints[i], breakpoints[i + 1]) for i in range(len(breakpoints) - 1)]

        Phi_atoms: List[np.ndarray] = []
        for low, high in atoms:
            mid = (low + high) / 2
            Phi_atoms.append(np.array([1 if g[0] <= mid < g[1] else 0 for g in self.groups], dtype=float))

        # Local digests per client and atom
        local_tdigests: List[List[TDigest]] = []
        for k in range(self.K):
            X_k, S_k = X_calib[k], S_calib[k]
            w_k_const = float(w_calib[k][0]) if len(w_calib[k]) > 0 else 0.0
            atom_tdigests_k: List[TDigest] = []
            for low, high in atoms:
                mask = (X_k >= low) & (X_k < high)
                td = TDigest(delta=delta, K=self.tdigest_K)
                if np.sum(mask) > 0:
                    for s in S_k[mask]:
                        td.update(float(s), w_k_const)
                atom_tdigests_k.append(td)
            local_tdigests.append(atom_tdigests_k)

        pseudo_S: List[float] = []
        pseudo_w: List[float] = []
        pseudo_Phi: List[np.ndarray] = []

        for a in range(len(atoms)):
            merged = TDigest(delta=delta, K=self.tdigest_K)
            for k in range(self.K):
                merged = merged + local_tdigests[k][a]
            for c in merged.centroids_to_list():
                pseudo_S.append(float(c['m']))
                pseudo_w.append(float(c['c']))
                pseudo_Phi.append(Phi_atoms[a])

        return np.array(pseudo_S, dtype=float), np.array(pseudo_w, dtype=float), np.array(pseudo_Phi, dtype=float)

    @staticmethod
    def find_S_star_gcfcp(
        X_test_point: float,
        pseudo_S: np.ndarray,
        pseudo_w: np.ndarray,
        pseudo_Phi: np.ndarray,
        w_test: float,
        alpha: float,
        groups: List[List[int]],
    ) -> float:
        """Solve the GC-FCP dual LP on pseudo-data by bisection."""
        d = len(groups)
        Phi_test = np.array([1 if g[0] <= X_test_point < g[1] else 0 for g in groups], dtype=float)

        m = len(pseudo_S)
        A_eq = np.hstack((pseudo_Phi.T, Phi_test.reshape(-1, 1)))
        b_eq = np.zeros(d)
        bounds = [(-pseudo_w[i] * alpha, pseudo_w[i] * (1 - alpha)) for i in range(m)]
        bounds.append((-w_test * alpha, w_test * (1 - alpha)))

        low, high = 0.0, 1.0
        U = (1 - alpha)
        for _ in range(50):
            mid = (low + high) / 2
            c = np.append(-pseudo_S, -mid)
            res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            if not res.success:
                return float(low)
            eta_test_val = res.x[-1]
            if eta_test_val >= w_test * U:
                high = mid
            else:
                low = mid
            if high - low < 1e-8:
                break
        return float(high)

    # -----------------
    # Metrics
    # -----------------
    def _miscoverage_and_setsize(
        self,
        X_test: np.ndarray,
        S_test: np.ndarray,
        probs_test: List[np.ndarray],
        tau_or_taus: np.ndarray,
        per_point: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return vectors (miscov, set_size) of length 1 + |groups|."""
        n_test = len(X_test)
        miscov = np.zeros(1 + len(self.groups))
        setsz = np.zeros(1 + len(self.groups))

        if per_point:
            taus = tau_or_taus
            miscov[0] = float(np.mean(S_test > taus))
            set_sizes = np.array([np.sum(probs_test[i] >= 1 - taus[i]) for i in range(n_test)], dtype=float)
            setsz[0] = float(np.mean(set_sizes))
        else:
            tau = float(tau_or_taus)
            miscov[0] = float(np.mean(S_test > tau))
            set_sizes = np.array([np.sum(probs_test[i] >= 1 - tau) for i in range(n_test)], dtype=float)
            setsz[0] = float(np.mean(set_sizes))

        for g, (low_b, up_b) in enumerate(self.groups):
            mask = (X_test >= low_b) & (X_test < up_b)
            if np.sum(mask) == 0:
                miscov[g + 1] = np.nan
                setsz[g + 1] = np.nan
                continue
            if per_point:
                miscov[g + 1] = float(np.mean(S_test[mask] > taus[mask]))
                setsz[g + 1] = float(np.mean(set_sizes[mask]))
            else:
                miscov[g + 1] = float(np.mean(S_test[mask] > tau))
                setsz[g + 1] = float(np.mean(set_sizes[mask]))

        return miscov, setsz

    # -----------------
    # Core run
    # -----------------
    def run(self) -> None:
        """Run Monte Carlo experiments and populate avg_* results."""
        miscov_mc: Dict[str, List[np.ndarray]] = {m: [] for m in self.methods}
        setsz_mc: Dict[str, List[np.ndarray]] = {m: [] for m in self.methods}
        time_mc: Dict[str, List[float]] = {m: [] for m in self.methods}
        self.run_seeds = []

        for mc_idx in tqdm(range(self.num_mc)):
            seed = self.base_seed + mc_idx
            self.run_seeds.append(seed)
            self._set_all_seeds(seed)

            # ----- load data and split 50/50 -----
            transform = transforms.ToTensor()
            if self.dataset == 'cifar10':
                ds = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
                get_item = lambda idx: ds[idx]
                N = len(ds)
            elif self.dataset == 'pathmnist':
                items = self._load_pathmnist_items()
                get_item = lambda idx: items[idx]
                N = len(items)
            else:
                raise ValueError(f"Unknown dataset: {self.dataset}")

            indices = list(range(N))
            random.shuffle(indices)
            split = N // 2
            calib_indices = indices[:split]
            eval_indices = indices[split:]

            calib_data = [get_item(i) for i in calib_indices]
            eval_data = [get_item(i) for i in eval_indices]

            # ----- assign calibration to clients by labels -----
            client_calib_images = [[] for _ in range(self.K)]
            client_calib_labels = [[] for _ in range(self.K)]
            for img, label in calib_data:
                label = int(label)
                for k in range(self.K):
                    if label in self.client_label_sets[k]:
                        client_calib_images[k].append(img)
                        client_calib_labels[k].append(label)
                        break

            n_k = np.array([len(x) for x in client_calib_images], dtype=int)
            lambda_k = [self.pi_k[k] / (n_k[k] + 1) for k in range(self.K)]
            w_test = float(np.sum(lambda_k))

            # ----- compute calibration scores/preds per client -----
            calib_scores: List[np.ndarray] = []
            calib_preds: List[np.ndarray] = []
            for k in range(self.K):
                scores_k, preds_k, _ = self.compute_scores_and_preds(client_calib_images[k], client_calib_labels[k])
                calib_scores.append(scores_k)
                calib_preds.append(preds_k)

            # ----- evaluation -----
            test_images = [img for img, _ in eval_data]
            test_labels = [int(y) for _, y in eval_data]
            S_test, X_test, probs_test = self.compute_scores_and_preds(test_images, test_labels)
            n_test = len(test_images)

            # ----- bootstrap calibration within each client -----
            X_calib: List[np.ndarray] = []
            S_calib: List[np.ndarray] = []
            w_calib: List[np.ndarray] = []
            for k in range(self.K):
                if n_k[k] == 0:
                    X_calib.append(np.array([], dtype=float))
                    S_calib.append(np.array([], dtype=float))
                    w_calib.append(np.array([], dtype=float))
                    continue
                idxs = np.random.choice(n_k[k], n_k[k], replace=True)
                X_k = calib_preds[k][idxs].astype(float)
                S_k = calib_scores[k][idxs].astype(float)
                X_calib.append(X_k)
                S_calib.append(S_k)
                w_calib.append(np.full(len(X_k), lambda_k[k], dtype=float))

            X_all = np.concatenate(X_calib).astype(float)
            S_all = np.concatenate(S_calib).astype(float)
            w_all = np.concatenate(w_calib).astype(float)

            # ----------------------
            # Global-threshold methods
            # ----------------------
            tau_cp = self.compute_vanilla_quantile(S_calib)
            tau_fedcp = self.compute_fedcp_quantile(S_calib, lambda_k)

            miscov, setsz = self._miscoverage_and_setsize(X_test, S_test, probs_test, np.array(tau_cp), per_point=False)
            miscov_mc['Centralized CP'].append(miscov)
            setsz_mc['Centralized CP'].append(setsz)
            time_mc['Centralized CP'].append(float('nan'))

            miscov, setsz = self._miscoverage_and_setsize(X_test, S_test, probs_test, np.array(tau_fedcp), per_point=False)
            miscov_mc['FedCP'].append(miscov)
            setsz_mc['FedCP'].append(setsz)
            time_mc['FedCP'].append(float('nan'))

            # ----------------------
            # Per-test-point methods
            # ----------------------
            n_all = len(X_all)
            w_all_condcp = np.full(n_all, 1.0 / (n_all + 1), dtype=float)
            w_test_condcp = 1.0 / (n_all + 1)

            with Pool(self.n_jobs) as pool:
                # Centralized CondCP
                # t0 = time.time()
                # args_condcp = [(float(x), X_all, S_all, w_all_condcp, w_test_condcp, self.ALPHA, self.groups) for x in X_test]
                # taus_condcp = np.array(pool.starmap(TableExperiment.find_S_star, args_condcp), dtype=float)
                # t1 = time.time()
                # t_condcp = (t1 - t0) / max(n_test, 1)
                #
                # miscov, setsz = self._miscoverage_and_setsize(X_test, S_test, probs_test, taus_condcp, per_point=True)
                # miscov_mc['Centralized CondCP'].append(miscov)
                # setsz_mc['Centralized CondCP'].append(setsz)
                # time_mc['Centralized CondCP'].append(float(t_condcp))

                # Centralized GC-FCP
                # t0 = time.time()
                # args_cg = [(float(x), X_all, S_all, w_all, w_test, self.ALPHA, self.groups) for x in X_test]
                # taus_cg = np.array(pool.starmap(TableExperiment.find_S_star, args_cg), dtype=float)
                # t1 = time.time()
                # t_cg = (t1 - t0) / max(n_test, 1)
                #
                # miscov, setsz = self._miscoverage_and_setsize(X_test, S_test, probs_test, taus_cg, per_point=True)
                # miscov_mc['Centralized GC-FCP'].append(miscov)
                # setsz_mc['Centralized GC-FCP'].append(setsz)
                # time_mc['Centralized GC-FCP'].append(float(t_cg))

                # GC-FCP (δ variants)
                for delta in self.gcfcp_deltas:
                    method_name = f'GC-FCP (δ={self._fmt_delta(delta)})'

                    # Offline pseudo-data (not included in per-test timing)
                    pseudo_S, pseudo_w, pseudo_Phi = self.compute_gcfcp_pseudo_data_fed(X_calib, S_calib, w_calib, delta=float(delta))

                    t0 = time.time()
                    args_g = [(float(x), pseudo_S, pseudo_w, pseudo_Phi, w_test, self.ALPHA, self.groups) for x in X_test]
                    taus_g = np.array(pool.starmap(TableExperiment.find_S_star_gcfcp, args_g), dtype=float)
                    t1 = time.time()
                    t_g = (t1 - t0) / max(n_test, 1)

                    miscov, setsz = self._miscoverage_and_setsize(X_test, S_test, probs_test, taus_g, per_point=True)
                    miscov_mc[method_name].append(miscov)
                    setsz_mc[method_name].append(setsz)
                    time_mc[method_name].append(float(t_g))

            print(f"[MC {mc_idx + 1:>3}/{self.num_mc}] done")

        # ----- retain raw results and aggregate -----
        for m in self.methods:
            self.raw_miscov[m] = np.stack(miscov_mc[m], axis=0).astype(float)
            self.raw_set_size[m] = np.stack(setsz_mc[m], axis=0).astype(float)
            self.raw_time_per_test[m] = np.array(time_mc[m], dtype=float)

            self.avg_miscov[m] = np.nanmean(self.raw_miscov[m], axis=0)
            self.avg_set_size[m] = np.nanmean(self.raw_set_size[m], axis=0)
            self.avg_miscov_stderr[m] = self._stderr(self.raw_miscov[m], axis=0)
            self.avg_set_size_stderr[m] = self._stderr(self.raw_set_size[m], axis=0)

            arr = self.raw_time_per_test[m]
            if np.isfinite(arr).any():
                self.avg_time_per_test[m] = float(np.nanmean(arr))
                self.avg_time_per_test_stderr[m] = float(self._stderr(arr, axis=0))
            else:
                self.avg_time_per_test[m] = None
                self.avg_time_per_test_stderr[m] = None

    # -----------------
    # Table formatting
    # -----------------
    def to_table_dataframe(self) -> pd.DataFrame:
        """Build the markdown-ready results table with mean +- standard error."""
        rows: List[Dict[str, str]] = []

        t_base = self.avg_time_per_test.get('Centralized CondCP', None)

        for method in self.methods:
            row: Dict[str, str] = {'Methods': method}

            cov_marg = 1.0 - float(self.avg_miscov[method][0])
            cov_marg_se = float(self.avg_miscov_stderr[method][0])
            row['Marginal Coverage'] = self._fmt_mean_se(cov_marg, cov_marg_se, digits=3)

            for gi in range(len(self.groups)):
                cov_g = 1.0 - float(self.avg_miscov[method][gi + 1])
                cov_g_se = float(self.avg_miscov_stderr[method][gi + 1])
                sz_g = float(self.avg_set_size[method][gi + 1])
                sz_g_se = float(self.avg_set_size_stderr[method][gi + 1])
                row[f"Coverage (and Set size) on G{gi + 1}"] = (
                    f"{cov_g:.3f} +- {cov_g_se:.3f} ({sz_g:.2f} +- {sz_g_se:.2f})"
                )

            if method in ['Centralized CP', 'FedCP']:
                row['Computational Acceleration*'] = 'N/A'
            elif method == 'Centralized CondCP':
                row['Computational Acceleration*'] = '(Serve as baseline)'
            else:
                t_m = self.avg_time_per_test.get(method, None)
                if t_base is None or t_m is None or t_m <= 0:
                    row['Computational Acceleration*'] = 'N/A'
                else:
                    row['Computational Acceleration*'] = f"{t_base / t_m:.2f}x"

            rows.append(row)

        cols = ['Methods', 'Marginal Coverage'] + [f"Coverage (and Set size) on G{i + 1}" for i in range(len(self.groups))] + [
            'Computational Acceleration*'
        ]
        return pd.DataFrame(rows)[cols]

    def to_raw_results_dataframe(self) -> pd.DataFrame:
        """Return all per-MC numerical table metrics in long format.

        Each row is one Monte Carlo repetition, one method, and one metric scope
        (marginal or group). This CSV is intended as the canonical input for
        later plotting/table generation without re-running the experiments.
        """
        rows: List[Dict[str, Any]] = []
        specs = self._metric_specs()
        for method in self.methods:
            if method not in self.raw_miscov or method not in self.raw_set_size:
                continue
            times = self.raw_time_per_test.get(method, np.full(self.raw_miscov[method].shape[0], np.nan))
            for mc_idx in range(self.raw_miscov[method].shape[0]):
                seed = self.run_seeds[mc_idx] if mc_idx < len(self.run_seeds) else self.base_seed + mc_idx
                for metric_idx, scope, label, low_b, up_b in specs:
                    miscov = float(self.raw_miscov[method][mc_idx, metric_idx])
                    set_size = float(self.raw_set_size[method][mc_idx, metric_idx])
                    rows.append({
                        'dataset': self.dataset,
                        'overlap': self.overlap,
                        'alpha': self.ALPHA,
                        'mc_idx': mc_idx,
                        'seed': seed,
                        'method': method,
                        'scope': scope,
                        'group': label,
                        'group_index': metric_idx - 1 if scope == 'group' else -1,
                        'group_low': low_b,
                        'group_high': up_b,
                        'coverage': 1.0 - miscov if np.isfinite(miscov) else np.nan,
                        'miscoverage': miscov,
                        'set_size': set_size,
                        'time_per_test': float(times[mc_idx]) if mc_idx < len(times) else np.nan,
                    })
        return pd.DataFrame(rows)

    def to_summary_stats_dataframe(self) -> pd.DataFrame:
        """Return mean and standard error of every reported numerical metric."""
        rows: List[Dict[str, Any]] = []
        specs = self._metric_specs()
        for method in self.methods:
            time_mean = self.avg_time_per_test.get(method, None)
            time_se = self.avg_time_per_test_stderr.get(method, None)
            for metric_idx, scope, label, low_b, up_b in specs:
                mis_mean = float(self.avg_miscov[method][metric_idx])
                mis_se = float(self.avg_miscov_stderr[method][metric_idx])
                sz_mean = float(self.avg_set_size[method][metric_idx])
                sz_se = float(self.avg_set_size_stderr[method][metric_idx])
                rows.append({
                    'dataset': self.dataset,
                    'overlap': self.overlap,
                    'alpha': self.ALPHA,
                    'num_mc': self.num_mc,
                    'method': method,
                    'scope': scope,
                    'group': label,
                    'group_index': metric_idx - 1 if scope == 'group' else -1,
                    'group_low': low_b,
                    'group_high': up_b,
                    'coverage_mean': 1.0 - mis_mean,
                    'coverage_stderr': mis_se,
                    'miscoverage_mean': mis_mean,
                    'miscoverage_stderr': mis_se,
                    'set_size_mean': sz_mean,
                    'set_size_stderr': sz_se,
                    'time_per_test_mean': time_mean if time_mean is not None else np.nan,
                    'time_per_test_stderr': time_se if time_se is not None else np.nan,
                })
        return pd.DataFrame(rows)

    def print_markdown_table(self) -> None:
        df = self.to_table_dataframe()
        print("\n" + "=" * 110)
        print(f"Results table for dataset={self.dataset}, overlap={self.overlap}, alpha={self.ALPHA}")
        print("=" * 110)
        print(df.to_markdown(index=False))
        print("\nCoverage and set size are reported as mean +- standard error over Monte Carlo runs.")
        print("*Acceleration baseline: Centralized CondCP; acceleration = AvgTime_CondCP / AvgTime_Method.")
        print("  Centralized CP and FedCP compute a single global threshold and are excluded (N/A).")

    def save_table(self, out_prefix: str) -> None:
        """Save formatted table, raw per-MC results, and summary statistics."""
        df = self.to_table_dataframe()
        raw_df = self.to_raw_results_dataframe()
        summary_df = self.to_summary_stats_dataframe()

        csv_path = f"{out_prefix}.csv"
        md_path = f"{out_prefix}.md"
        json_path = f"{out_prefix}.json"
        raw_csv_path = f"{out_prefix}_raw_results.csv"
        summary_csv_path = f"{out_prefix}_summary_stats.csv"
        raw_npz_path = f"{out_prefix}_raw_arrays.npz"

        df.to_csv(csv_path, index=False)
        raw_df.to_csv(raw_csv_path, index=False)
        summary_df.to_csv(summary_csv_path, index=False)

        with open(md_path, 'w') as f:
            f.write(df.to_markdown(index=False))
            f.write("\n\nAll coverage and set-size entries are reported as mean +- standard error over Monte Carlo repetitions.\n")
            f.write("\n*Acceleration baseline: Centralized CondCP; acceleration = AvgTime_CondCP / AvgTime_Method.\n")

        arrays: Dict[str, np.ndarray] = {
            'groups': np.array(self.groups, dtype=float),
            'run_seeds': np.array(self.run_seeds, dtype=int),
        }
        for method in self.methods:
            key = self._safe_key(method)
            arrays[f'{key}__miscov'] = self.raw_miscov[method]
            arrays[f'{key}__set_size'] = self.raw_set_size[method]
            arrays[f'{key}__time_per_test'] = self.raw_time_per_test[method]
        np.savez_compressed(raw_npz_path, **arrays)

        payload = {
            'dataset': self.dataset,
            'overlap': self.overlap,
            'alpha': self.ALPHA,
            'groups': self.groups,
            'eval_ranges': self.eval_ranges,
            'methods': self.methods,
            'num_mc': self.num_mc,
            'run_seeds': self.run_seeds,
            'raw_miscov': {k: v.tolist() for k, v in self.raw_miscov.items()},
            'raw_set_size': {k: v.tolist() for k, v in self.raw_set_size.items()},
            'raw_time_per_test': {k: v.tolist() for k, v in self.raw_time_per_test.items()},
            'avg_miscov': {k: v.tolist() for k, v in self.avg_miscov.items()},
            'avg_miscov_stderr': {k: v.tolist() for k, v in self.avg_miscov_stderr.items()},
            'avg_set_size': {k: v.tolist() for k, v in self.avg_set_size.items()},
            'avg_set_size_stderr': {k: v.tolist() for k, v in self.avg_set_size_stderr.items()},
            'avg_time_per_test': self.avg_time_per_test,
            'avg_time_per_test_stderr': self.avg_time_per_test_stderr,
            'table': df.to_dict(orient='records'),
            'raw_results_csv': raw_csv_path,
            'summary_stats_csv': summary_csv_path,
            'raw_arrays_npz': raw_npz_path,
            'timestamp': datetime.now().isoformat(),
        }
        with open(json_path, 'w') as f:
            json.dump(payload, f, indent=2)

        print(f"Saved formatted table: {csv_path}")
        print(f"Saved formatted table: {md_path}")
        print(f"Saved raw per-MC results: {raw_csv_path}")
        print(f"Saved summary statistics: {summary_csv_path}")
        print(f"Saved raw arrays: {raw_npz_path}")
        print(f"Saved metadata/results JSON: {json_path}")


def _parse_deltas(s: str) -> List[float]:
    out: List[float] = []
    for part in (s or '').split(','):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    if not out:
        raise ValueError("Empty --gcfcp_deltas. Provide e.g. '1,0.1,0.01'.")
    return out


def main_table(args: argparse.Namespace) -> None:
    deltas = _parse_deltas(args.gcfcp_deltas)

    exp = TableExperiment(
        dataset=args.dataset,
        overlap=args.overlap,
        K=args.clients,
        num_mc=args.times,
        n_jobs=args.n_jobs,
        gcfcp_deltas=deltas,
        tdigest_K=args.tdigest_K,
        model_path=args.model_path,
        base_seed=args.base_seed,
    )

    exp.run()
    exp.print_markdown_table()

    stamp = datetime.now().strftime('%Y%m%d')
    out_prefix = os.path.join(
        'results',
        f"table_{exp.dataset}_alpha{exp.ALPHA}_overlap{int(exp.overlap)}_mc{exp.num_mc}_{stamp}",
    )
    exp.save_table(out_prefix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GC-FCP real-data table experiment (alpha fixed to 0.1).')
    parser.add_argument('--times', type=int, default=50, help='Monte Carlo runs (random splits + bootstrap).')
    parser.add_argument('--clients', type=int, default=5, help='Number of clients.')
    parser.add_argument('--n_jobs', type=int, default=5, help='Parallel jobs for per-test LP solves (-1 uses all cores).')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'pathmnist'],
                        help='Dataset for the experiment.')

    # Default matches the user-provided table layout (overlapping groups)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--overlap', dest='overlap', action='store_true', help='Use overlapping groups (default).')
    group.add_argument('--disjoint', dest='overlap', action='store_false', help='Use disjoint groups.')
    parser.set_defaults(overlap=True)

    parser.add_argument('--model_path', type=str, default='checkpoints/path_cnn.pt',
                        help='Path to PathMNIST checkpoint (PathMNIST only).')
    parser.add_argument('--gcfcp_deltas', type=str, default='1,0.1,0.01',
                        help='Comma-separated GC-FCP 1/δ values (e.g., "1,0.1,0.01").')
    parser.add_argument('--tdigest_K', type=int, default=25, help='Internal TDigest parameter K (library-specific).')
    parser.add_argument('--base_seed', type=int, default=114514, help='Base RNG seed.')

    args = parser.parse_args()
    os.makedirs('results', exist_ok=True)
    main_table(args)

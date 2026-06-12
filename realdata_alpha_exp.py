import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import linprog
import pandas as pd
from multiprocessing import Pool, cpu_count
from functools import partial
from typing import Tuple, List, Dict, Any
import warnings
from tqdm import tqdm
import time
from tdigest import TDigest
import torch
import torchvision
import torchvision.transforms as transforms
import torch.hub
import os
import random
import matplotlib as mpl
import pickle
import json
from datetime import datetime
from typing import Callable

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
torch.hub.set_dir('./model')

warnings.filterwarnings('ignore')

plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
font = {'family': 'Times New Roman',
        # 'weight': 'bold',
        'size': 16}

mpl.rc('font', **font)

# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)


class Experiment:
    """
    A class for running conformal prediction experiments on CIFAR10 with overlapping groups.

    This experiment compares vanilla conformal prediction, a centralized_gcfcp dual LP approach, and GC-FCP
    for conditional conformal prediction across overlapping groups.
    """

    def __init__(self,
                 K: int = 5,
                 pi_k: List[float] = None,
                 alpha: float = 0.1,
                 num_mc: int = 100,
                 n_jobs: int = -1,
                 delta: float = 0.01,
                 compression: int = 25,
                 overlap: bool = False,
                 run_centralized_gcfcp: bool = False,
                 dataset: str = 'cifar10',   # dataset ∈ {cifar10, pathmnist}
                 model_path: str = None,
                 base_seed: int = 114514):
        """
        Initialize the experiment parameters.

        Args:
            K: Number of clients (superclasses in CIFAR100)
            alpha: Significance level
            num_mc: Number of Monte Carlo simulations
            n_jobs: Number of parallel jobs (-1 for all CPUs)
            compression: Compression factor δ for GC-FCP (higher = more accurate, more points)
            run_centralized_gcfcp: Flag to control whether to run the centralized_gcfcp approach
        """
        self.K = K
        self.pi_k = pi_k if pi_k is not None else [1 / K] * K
        self.alpha = alpha
        self.num_mc = num_mc
        self.n_jobs = n_jobs if n_jobs != -1 else cpu_count()
        self.compression = compression
        self.delta = delta
        self.run_centralized_gcfcp = run_centralized_gcfcp
        self.overlap = overlap
        self.loaded = False
        self.base_seed = int(base_seed)

        # ---------- Dataset-specific init ----------
        self.dataset = (dataset or 'cifar10').lower()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if self.dataset == 'cifar10':
            # Pretrained CIFAR-10 ResNet56
            self.model = torch.hub.load("chenyaofo/pytorch-cifar-models",
                                        "cifar10_resnet56", pretrained=True)
            self.model.to(self.device).eval()
            self.normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                                  std=[0.2675, 0.2565, 0.2761])
            self.n_classes = 10

            # Groups identical to original script (overlap flag preserved)
            if self.overlap:
                self.groups = [[0, 4], [2, 6], [4, 8], [6, 10]]
                self.eval_ranges = ['{0,1,2,3}', '{2,3,4,5}', '{4,5,6,7}', '{6,7,8,9}']
            else:
                self.groups = [[0, 2], [2, 4], [4, 6], [6, 8], [8, 10]]
                self.eval_ranges = ['{0,1}', '{2,3}', '{4,5}', '{6,7}', '{8,9}']

            # Client label partitions: equal split of 10 classes
            per = int(self.n_classes / self.K)
            all_labels = list(range(self.n_classes))
            self.client_label_sets = [set(all_labels[i*per:(i+1)*per]) for i in range(self.K)]

        elif self.dataset == 'pathmnist':
            # Lazy imports to avoid dependency for CIFAR runs
            from medmnist import INFO
            import medmnist
            try:
                from medmnist_train import SimpleCNN  # same class used in your medmnist demo
            except Exception as e:
                raise ImportError("SimpleCNN not found. Ensure medmnist_train.py is available.") from e

            # 9-way SimpleCNN; load checkpoint if provided
            self.model = SimpleCNN(n_classes=9).to(self.device)
            if model_path is not None:
                ckpt = torch.load(model_path, map_location=self.device)
                if isinstance(ckpt, dict) and 'model_state' in ckpt:
                    self.model.load_state_dict(ckpt['model_state'])
                elif isinstance(ckpt, dict):
                    self.model.load_state_dict(ckpt)
                else:
                    self.model = ckpt.to(self.device)
            self.model.eval()

            # Identity normalization (your PathMNIST model expects 0-1 tensors)
            self.normalize = (lambda x: x)
            self.n_classes = 9

            # Group intervals as in your MedMNIST demo (overlapping by predicted class)
            # (If you also want a disjoint option, you can keep the 'else' branch below.)
            if self.overlap:
                self.groups = [[0, 3], [1, 4], [2, 5], [3, 6], [4, 9]]
                self.eval_ranges = ['{0,1,2}', '{1,2,3}', '{2,3,4}', '{3,4,5}', '{4,5,6,7,8}']
            else:
                # optional disjoint fallback
                self.groups = [[0, 2], [2, 4], [4, 6], [6, 8], [8, 9]]
                self.eval_ranges = ['{0,1}', '{2,3}', '{4,5}', '{6,7}', '{8}']

            # Client label holdings as in demo
            self.client_label_sets = [ {0,1}, {2,3}, {4,5}, {6,7}, {8} ]

        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")

        self.d = len(self.groups)

        # Results storage
        self.miscov_vanilla_mc = None
        self.miscov_centralized_gcfcp_mc = None
        self.miscov_gcfcp_mc = None
        self.set_size_vanilla_mc = None
        self.set_size_centralized_gcfcp_mc = None
        self.set_size_gcfcp_mc = None
        self.avg_miscov_vanilla = None
        self.avg_miscov_centralized_gcfcp = None
        self.avg_miscov_gcfcp = None
        self.avg_set_size_vanilla = None
        self.avg_set_size_centralized_gcfcp = None
        self.avg_set_size_gcfcp = None

    def _set_all_seeds(self, seed: int) -> None:
        """Seed Python, NumPy, PyTorch (CPU & CUDA) and make cuDNN deterministic."""
        self.current_seed = int(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)  # hash-based ops
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # cuDNN determinism
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # If available (PyTorch>=1.8), enforce deterministic ops
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

    def _load_pathmnist_items(self) -> List[Tuple[torch.Tensor, int]]:
        """Return a Python list of (tensor, intlabel) from PathMNIST val+test."""
        from medmnist import INFO
        import medmnist
        transform = transforms.Compose([transforms.ToTensor()])
        data_flag = 'pathmnist'
        info = INFO[data_flag]
        DataClass = getattr(medmnist, info['python_class'])
        val_ds  = DataClass(split='val',  transform=transform, download=True, root='data')
        test_ds = DataClass(split='test', transform=transform, download=True, root='data')

        def _extract(ds):
            out = []
            for i in range(len(ds)):
                x, y = ds[i]
                # Robust label to int (handles scalar, [[k]], one-hot, tensor)
                if isinstance(y, np.ndarray):
                    y_int = int(y) if y.ndim == 0 else (int(y.item()) if (y.ndim == 1 and y.size == 1) else int(np.argmax(y)))
                elif torch.is_tensor(y):
                    y_int = int(y.item()) if y.ndim <= 1 else int(torch.argmax(y).item())
                else:
                    y_int = int(y)
                out.append((x, y_int))
            return out

        return _extract(val_ds) + _extract(test_ds)

    def compute_scores_and_preds(self, images: List[torch.Tensor], labels: List[int]) -> Tuple[
        np.ndarray, np.ndarray, List[np.ndarray]]:
        """Compute nonconformity scores: 1 - prob_y and predicted classes."""
        scores = []
        preds = []
        probs = []
        batch_size = 128
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch_x = torch.stack(images[i:i + batch_size]).to(self.device)
                batch_x = self.normalize(batch_x)
                batch_y = torch.tensor(labels[i:i + batch_size]).to(self.device)
                logits = self.model(batch_x)
                soft = torch.softmax(logits, dim=1)
                prob_y = soft.gather(1, batch_y.unsqueeze(1)).squeeze(1).cpu().numpy()
                scores.extend(1 - prob_y)
                batch_preds = logits.argmax(1).cpu().numpy()
                preds.extend(batch_preds)
                batch_probs = soft.cpu().numpy()
                probs.extend(batch_probs)
        return np.array(scores), np.array(preds), probs

    def compute_vanilla_quantiles(self, S_calib: List[np.ndarray]) -> float:
        """Compute vanilla conformal prediction quantile."""
        S_calib_pooled = np.concatenate(S_calib)
        n = len(S_calib_pooled)
        q_high = np.quantile(S_calib_pooled, (1 - self.alpha) * (1 + 1 / n))
        return q_high

    @staticmethod
    def find_S_star(X_test_point: float, X_all: np.ndarray, S_all: np.ndarray,
                    w_all: np.ndarray, w_test: float, alpha: float, groups: List[List[int]]) -> float:
        d = len(groups)
        Phi_test = np.array([1 if g[0] <= X_test_point < g[1] else 0 for g in groups])

        n = len(X_all)
        Phi_all = np.zeros((n, d))
        for i in range(n):
            Phi_all[i] = [1 if g[0] <= X_all[i] < g[1] else 0 for g in groups]

        A_eq = np.hstack((Phi_all.T, Phi_test.reshape(-1, 1)))
        b_eq = np.zeros(d)
        bounds = [(-w_all[i] * alpha, w_all[i] * (1 - alpha)) for i in range(n)]
        bounds.append((-w_test * alpha, w_test * (1 - alpha)))

        low = 0.0
        high = 1.0
        for _ in range(50):
            mid = (low + high) / 2
            c = np.append(-S_all, -mid)
            res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            if not res.success:
                return low
            eta_test_val = res.x[-1]
            if eta_test_val >= w_test * (1 - alpha):
                high = mid
            else:
                low = mid
            if high - low < 1e-8:
                break
        return high

    def compute_gcfcp_pseudo_data_fed(self, X_calib: List[np.ndarray], S_calib: List[np.ndarray],
                                   w_calib: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        breakpoints = sorted(set(sum(self.groups, [])))
        atoms = [(breakpoints[i], breakpoints[i + 1]) for i in range(len(breakpoints) - 1)]
        num_atoms = len(atoms)

        Phi_atoms = []
        for low, high in atoms:
            mid = (low + high) / 2
            Phi_atoms.append(np.array([1 if g[0] <= mid < g[1] else 0 for g in self.groups]))

        local_tdigests = []
        for k in range(self.K):
            X_k = X_calib[k]
            S_k = S_calib[k]
            w_k_const = w_calib[k][0] if len(w_calib[k]) > 0 else 0

            atom_tdigests_k = []
            for low, high in atoms:
                mask = (X_k >= low) & (X_k < high)
                td = TDigest(delta=self.delta, K=self.compression)
                if np.sum(mask) > 0:
                    S_j = S_k[mask]
                    for S in S_j:
                        td.update(S, w_k_const)
                atom_tdigests_k.append(td)
            local_tdigests.append(atom_tdigests_k)

        pseudo_S = []
        pseudo_w = []
        pseudo_Phi = []

        for a in range(num_atoms):
            merged = TDigest(delta=self.delta, K=self.compression)
            for k in range(self.K):
                merged = merged + local_tdigests[k][a]
            centroids = merged.centroids_to_list()
            for c in centroids:
                pseudo_S.append(c['m'])
                pseudo_w.append(c['c'])
                pseudo_Phi.append(Phi_atoms[a])

        return np.array(pseudo_S), np.array(pseudo_w), np.array(pseudo_Phi)

    @staticmethod
    def find_S_star_gcfcp(X_test_point: float, pseudo_S: np.ndarray, pseudo_w: np.ndarray,
                       pseudo_Phi: np.ndarray, w_test: float, alpha: float, groups: List[List[int]]) -> float:
        d = len(groups)
        Phi_test = np.array([1 if g[0] <= X_test_point < g[1] else 0 for g in groups])

        m = len(pseudo_S)
        A_eq = np.hstack((pseudo_Phi.T, Phi_test.reshape(-1, 1)))
        b_eq = np.zeros(d)
        bounds = [(-pseudo_w[i] * alpha, pseudo_w[i] * (1 - alpha)) for i in range(m)]
        bounds.append((-w_test * alpha, w_test * (1 - alpha)))

        low = 0.0
        high = 1.0
        U = (1 - alpha)
        # U = np.random.uniform(-alpha, 1-alpha, 1)
        for _ in range(50):
            mid = (low + high) / 2
            c = np.append(-pseudo_S, -mid)
            res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            if not res.success:
                return low
            eta_test_val = res.x[-1]
            if eta_test_val >= w_test * U:
                high = mid
            else:
                low = mid
            if high - low < 1e-8:
                break
        return high

    def compute_prediction_intervals(self, X_test: np.ndarray, X_all: np.ndarray, S_all: np.ndarray,
                                     w_all: np.ndarray,
                                     pseudo_S: np.ndarray, pseudo_w: np.ndarray,
                                     pseudo_Phi: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute thresholds for centralized_gcfcp dual and GC-FCP."""
        n_test = len(X_test)

        taus_centralized_gcfcp = np.zeros(n_test) if self.run_centralized_gcfcp else None
        if self.run_centralized_gcfcp:
            start_time = time.time()
            with Pool(self.n_jobs) as pool:
                args = [(X_test[i], X_all, S_all, w_all, self.w_test, self.alpha, self.groups) for i in range(n_test)]
                taus_centralized_gcfcp = np.array(pool.starmap(Experiment.find_S_star, args))
            end_time = time.time()

        # GC-FCP
        start_time = time.time()
        with Pool(self.n_jobs) as pool:
            args = [(X_test[i], pseudo_S, pseudo_w, pseudo_Phi, self.w_test, self.alpha, self.groups) for i in
                    range(n_test)]
            taus_gcfcp = np.array(pool.starmap(Experiment.find_S_star_gcfcp, args))
        end_time = time.time()

        return {
            'centralized_gcfcp': taus_centralized_gcfcp,
            'gcfcp': taus_gcfcp
        }

    def run_single_simulation(self, mc_idx: int) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run a single Monte Carlo simulation with bootstrap."""
        # Bootstrap calibration data
        X_calib = []
        S_calib = []
        w_calib = []
        for k in range(self.K):
            indices = np.random.choice(self.n_k[k], self.n_k[k], replace=True)
            S_k = self.calib_scores[k][indices]
            X_k = self.calib_preds[k][indices]
            w_k = np.full(len(X_k), self.lambda_k[k])
            X_calib.append(X_k)
            S_calib.append(S_k)
            w_calib.append(w_k)

        # Pool for centralized_gcfcp
        X_all = np.concatenate(X_calib)
        S_all = np.concatenate(S_calib)
        w_all = np.concatenate(w_calib)

        # Compute GC-FCP pseudo-data
        pseudo_S, pseudo_w, pseudo_Phi = self.compute_gcfcp_pseudo_data_fed(X_calib, S_calib, w_calib)

        # Compute vanilla quantile
        tau_vanilla = self.compute_vanilla_quantiles(S_calib)

        # Test data (fixed)
        X_test = self.test_X
        S_test = self.test_S

        # Compute thresholds
        intervals = self.compute_prediction_intervals(X_test, X_all, S_all, w_all, pseudo_S, pseudo_w, pseudo_Phi)
        taus_centralized_gcfcp = intervals['centralized_gcfcp']
        taus_gcfcp = intervals['gcfcp']

        # Compute miscoverage: marginal + len(groups)
        miscov_vanilla = np.zeros(1 + len(self.groups))
        miscov_centralized_gcfcp = np.zeros(1 + len(self.groups)) if self.run_centralized_gcfcp else np.full(1 + len(self.groups), np.nan)
        miscov_gcfcp = np.zeros(1 + len(self.groups))

        # Marginal
        miscov_vanilla[0] = np.mean(S_test > tau_vanilla)
        if self.run_centralized_gcfcp:
            miscov_centralized_gcfcp[0] = np.mean(S_test > taus_centralized_gcfcp)
        miscov_gcfcp[0] = np.mean(S_test > taus_gcfcp)

        # Overlapping groups
        for g in range(len(self.groups)):
            low_b, up_b = self.groups[g]
            mask = (X_test >= low_b) & (X_test < up_b)
            if np.sum(mask) > 0:
                miscov_vanilla[g + 1] = np.mean(S_test[mask] > tau_vanilla)
                if self.run_centralized_gcfcp:
                    miscov_centralized_gcfcp[g + 1] = np.mean(S_test[mask] > taus_centralized_gcfcp[mask])
                miscov_gcfcp[g + 1] = np.mean(S_test[mask] > taus_gcfcp[mask])

        # Compute set sizes: marginal + len(groups)
        set_size_vanilla = np.zeros(1 + len(self.groups))
        set_size_centralized_gcfcp = np.zeros(1 + len(self.groups)) if self.run_centralized_gcfcp else np.full(1 + len(self.groups), np.nan)
        set_size_gcfcp = np.zeros(1 + len(self.groups))

        # Marginal
        set_sizes_v = np.array([np.sum(self.test_probs[i] >= 1 - tau_vanilla) for i in range(self.n_test)])
        set_size_vanilla[0] = np.mean(set_sizes_v)
        if self.run_centralized_gcfcp:
            set_sizes_n = np.array([np.sum(self.test_probs[i] >= 1 - taus_centralized_gcfcp[i]) for i in range(self.n_test)])
            set_size_centralized_gcfcp[0] = np.mean(set_sizes_n)
        set_sizes_g = np.array([np.sum(self.test_probs[i] >= 1 - taus_gcfcp[i]) for i in range(self.n_test)])
        set_size_gcfcp[0] = np.mean(set_sizes_g)

        # Overlapping groups
        for g in range(len(self.groups)):
            low_b, up_b = self.groups[g]
            mask = (X_test >= low_b) & (X_test < up_b)
            if np.sum(mask) > 0:
                set_size_vanilla[g + 1] = np.mean(set_sizes_v[mask])
                if self.run_centralized_gcfcp:
                    set_size_centralized_gcfcp[g + 1] = np.mean(set_sizes_n[mask])
                set_size_gcfcp[g + 1] = np.mean(set_sizes_g[mask])

        return miscov_vanilla, miscov_centralized_gcfcp, miscov_gcfcp, set_size_vanilla, set_size_centralized_gcfcp, set_size_gcfcp

    def run_monte_carlo(self) -> None:
        """Run Monte Carlo simulations serially."""
        print(f"Running {self.num_mc} Monte Carlo simulations serially...")
        self.seed_log = []

        results = []
        for mc_idx in tqdm(range(self.num_mc)):
            seed = self.base_seed + mc_idx
            self._set_all_seeds(seed)
            self.seed_log.append(seed)

            # Load & split data 50/50 depending on dataset
            transform = transforms.ToTensor()
            if self.dataset == 'cifar10':
                dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                       download=True, transform=transform)
                get_item = lambda idx: dataset[idx]
                N = len(dataset)
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
            eval_indices  = indices[split:]

            calib_data = [get_item(i) for i in calib_indices]
            eval_data  = [get_item(i) for i in eval_indices]

            # Split calibration across clients using dataset-specific label sets
            self.client_calib_images = [[] for _ in range(self.K)]
            self.client_calib_labels = [[] for _ in range(self.K)]
            for img, label in calib_data:
                for k in range(self.K):
                    if label in self.client_label_sets[k]:
                        self.client_calib_images[k].append(img)
                        self.client_calib_labels[k].append(label)
                        break

            self.n_k = np.array([len(c) for c in self.client_calib_images])
            # self.pi_k = [(self.n_k[k] + 1) / (np.sum(self.n_k) + self.K) for k in range(self.K)]

            # Precompute scores and predicted classes for calibration data
            self.calib_scores = [None] * self.K
            self.calib_preds = [None] * self.K
            for k in range(self.K):
                scores, preds, _ = self.compute_scores_and_preds(self.client_calib_images[k],
                                                                 self.client_calib_labels[k])
                self.calib_scores[k] = scores
                self.calib_preds[k] = preds

            # Precompute for eval data (test)
            self.test_images = [img for img, _ in eval_data]
            self.test_labels = [fine for _, fine in eval_data]
            self.test_S, self.test_X, self.test_probs = self.compute_scores_and_preds(self.test_images,
                                                                                      self.test_labels)
            self.n_test = len(self.test_images)

            # Derived parameters
            self.lambda_k = [self.pi_k[k] / (self.n_k[k] + 1) for k in range(self.K)]
            self.w_test = sum(self.lambda_k)

            results.append(self.run_single_simulation(mc_idx))

        miscov_vanilla_list, miscov_centralized_gcfcp_list, miscov_gcfcp_list, set_size_vanilla_list, set_size_centralized_gcfcp_list, set_size_gcfcp_list = zip(
            *results)

        self.miscov_vanilla_mc = np.array(miscov_vanilla_list)
        self.miscov_centralized_gcfcp_mc = np.array(miscov_centralized_gcfcp_list)
        self.miscov_gcfcp_mc = np.array(miscov_gcfcp_list)
        self.set_size_vanilla_mc = np.array(set_size_vanilla_list)
        self.set_size_centralized_gcfcp_mc = np.array(set_size_centralized_gcfcp_list)
        self.set_size_gcfcp_mc = np.array(set_size_gcfcp_list)

        self.avg_miscov_vanilla = np.mean(self.miscov_vanilla_mc, axis=0)
        self.avg_miscov_centralized_gcfcp = np.mean(self.miscov_centralized_gcfcp_mc, axis=0)
        self.avg_miscov_gcfcp = np.mean(self.miscov_gcfcp_mc, axis=0)
        self.avg_set_size_vanilla = np.mean(self.set_size_vanilla_mc, axis=0)
        self.avg_set_size_centralized_gcfcp = np.mean(self.set_size_centralized_gcfcp_mc, axis=0)
        self.avg_set_size_gcfcp = np.mean(self.set_size_gcfcp_mc, axis=0)

        print("Monte Carlo simulations completed.")

    def save_results(self, filename: str) -> None:
        """Save experiment results to file."""
        results_data = {
            'parameters': {
                'K': self.K,
                'pi_k': self.pi_k,
                'alpha': self.alpha,
                'num_mc': self.num_mc,
                'compression': self.compression,
                'delta': self.delta,
                'run_centralized_gcfcp': self.run_centralized_gcfcp,
                'overlap': self.overlap,
                'groups': self.groups,
                'dataset': self.dataset,
                'eval_ranges': self.eval_ranges
            },
            'results': {
                'miscov_vanilla_mc': self.miscov_vanilla_mc.tolist() if self.miscov_vanilla_mc is not None else None,
                'miscov_centralized_gcfcp_mc': self.miscov_centralized_gcfcp_mc.tolist() if self.miscov_centralized_gcfcp_mc is not None else None,
                'miscov_gcfcp_mc': self.miscov_gcfcp_mc.tolist() if self.miscov_gcfcp_mc is not None else None,
                'set_size_vanilla_mc': self.set_size_vanilla_mc.tolist() if self.set_size_vanilla_mc is not None else None,
                'set_size_centralized_gcfcp_mc': self.set_size_centralized_gcfcp_mc.tolist() if self.set_size_centralized_gcfcp_mc is not None else None,
                'set_size_gcfcp_mc': self.set_size_gcfcp_mc.tolist() if self.set_size_gcfcp_mc is not None else None,
                'avg_miscov_vanilla': self.avg_miscov_vanilla.tolist() if self.avg_miscov_vanilla is not None else None,
                'avg_miscov_centralized_gcfcp': self.avg_miscov_centralized_gcfcp.tolist() if self.avg_miscov_centralized_gcfcp is not None else None,
                'avg_miscov_gcfcp': self.avg_miscov_gcfcp.tolist() if self.avg_miscov_gcfcp is not None else None,
                'avg_set_size_vanilla': self.avg_set_size_vanilla.tolist() if self.avg_set_size_vanilla is not None else None,
                'avg_set_size_centralized_gcfcp': self.avg_set_size_centralized_gcfcp.tolist() if self.avg_set_size_centralized_gcfcp is not None else None,
                'avg_set_size_gcfcp': self.avg_set_size_gcfcp.tolist() if self.avg_set_size_gcfcp is not None else None
            },
            'timestamp': datetime.now().isoformat()
        }

        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"Results saved to {filename}")

    def load_results(self, filename: str) -> None:
        """Load experiment results from file."""
        with open(filename, 'r') as f:
            data = json.load(f)

        # Load parameters
        params = data['parameters']
        self.K = params['K']
        self.pi_k = params['pi_k']
        self.alpha = params['alpha']
        self.num_mc = params['num_mc']
        self.compression = params['compression']
        self.delta = params['delta']
        self.run_centralized_gcfcp = params['run_centralized_gcfcp']
        self.overlap = params['overlap']
        self.groups = params['groups']
        self.eval_ranges = params['eval_ranges']
        self.loaded = True
        self.dataset = params.get('dataset', 'cifar10')

        # Load results
        results = data['results']
        self.miscov_vanilla_mc = np.array(results['miscov_vanilla_mc']) if results[
                                                                               'miscov_vanilla_mc'] is not None else None
        self.miscov_centralized_gcfcp_mc = np.array(results['miscov_centralized_gcfcp_mc']) if results['miscov_centralized_gcfcp_mc'] is not None else None
        self.miscov_gcfcp_mc = np.array(results['miscov_gcfcp_mc']) if results['miscov_gcfcp_mc'] is not None else None
        self.set_size_vanilla_mc = np.array(results['set_size_vanilla_mc']) if results[
                                                                                   'set_size_vanilla_mc'] is not None else None
        self.set_size_centralized_gcfcp_mc = np.array(results['set_size_centralized_gcfcp_mc']) if results[
                                                                               'set_size_centralized_gcfcp_mc'] is not None else None
        self.set_size_gcfcp_mc = np.array(results['set_size_gcfcp_mc']) if results['set_size_gcfcp_mc'] is not None else None
        self.avg_miscov_vanilla = np.array(results['avg_miscov_vanilla']) if results[
                                                                                 'avg_miscov_vanilla'] is not None else None
        self.avg_miscov_centralized_gcfcp = np.array(results['avg_miscov_centralized_gcfcp']) if results[
                                                                             'avg_miscov_centralized_gcfcp'] is not None else None
        self.avg_miscov_gcfcp = np.array(results['avg_miscov_gcfcp']) if results['avg_miscov_gcfcp'] is not None else None
        self.avg_set_size_vanilla = np.array(results['avg_set_size_vanilla']) if results[
                                                                                     'avg_set_size_vanilla'] is not None else None
        self.avg_set_size_centralized_gcfcp = np.array(results['avg_set_size_centralized_gcfcp']) if results[
                                                                                 'avg_set_size_centralized_gcfcp'] is not None else None
        self.avg_set_size_gcfcp = np.array(results['avg_set_size_gcfcp']) if results['avg_set_size_gcfcp'] is not None else None

        print(f"Results loaded from {filename}")

    def generate_visualization_data(self, seed: int = 42) -> Dict[str, Any]:
        """Generate data for visualization with fixed seed (using one simulation)."""
        # np.random.seed(seed)

        # Use full calib for viz
        X_calib = []
        S_calib = []
        w_calib = []
        for k in range(self.K):
            S_k = self.calib_scores[k]
            X_k = self.calib_preds[k]
            w_k = np.full(len(X_k), self.lambda_k[k])
            X_calib.append(X_k)
            S_calib.append(S_k)
            w_calib.append(w_k)

        X_all = np.concatenate(X_calib)
        S_all = np.concatenate(S_calib)
        w_all = np.concatenate(w_calib)

        pseudo_S, pseudo_w, pseudo_Phi = self.compute_gcfcp_pseudo_data_fed(X_calib, S_calib, w_calib)

        tau_vanilla = self.compute_vanilla_quantiles(S_calib)
        taus_vanilla = np.full(self.n_test, tau_vanilla)

        intervals = self.compute_prediction_intervals(self.test_X, X_all, S_all, w_all, pseudo_S, pseudo_w, pseudo_Phi)
        taus_centralized_gcfcp = intervals['centralized_gcfcp']
        taus_gcfcp = intervals['gcfcp']

        return {
            'X_test': self.test_X,
            'S_test': self.test_S,
            'taus_vanilla': taus_vanilla,
            'taus_centralized_gcfcp': taus_centralized_gcfcp,
            'taus_gcfcp': taus_gcfcp
        }

    def create_coverage_dataframe(self) -> pd.DataFrame:
        """Create DataFrame for coverage visualization."""
        if self.avg_miscov_vanilla is None or self.avg_miscov_gcfcp is None:
            raise ValueError("Must run Monte Carlo simulations or load results first")

        coverage_data = [
            {'Range': 'Marginal', 'Miscoverage': self.avg_miscov_vanilla[0], 'Method': 'Centralized CP'},
        ]
        coverage_data.append({'Range': 'Marginal', 'Miscoverage': self.avg_miscov_gcfcp[0], 'Method': 'GC-FCP'})
        if self.run_centralized_gcfcp and self.avg_miscov_centralized_gcfcp is not None:
            coverage_data.append({'Range': 'Marginal', 'Miscoverage': self.avg_miscov_centralized_gcfcp[0], 'Method': 'CondDCP'})

        for r in range(len(self.eval_ranges)):
            coverage_data.append(
                {'Range': self.eval_ranges[r], 'Miscoverage': self.avg_miscov_vanilla[r + 1], 'Method': 'Centralized CP'})
            coverage_data.append(
                {'Range': self.eval_ranges[r], 'Miscoverage': self.avg_miscov_gcfcp[r + 1], 'Method': 'GC-FCP'})
            if self.run_centralized_gcfcp and self.avg_miscov_centralized_gcfcp is not None:
                coverage_data.append(
                    {'Range': self.eval_ranges[r], 'Miscoverage': self.avg_miscov_centralized_gcfcp[r + 1], 'Method': 'CondDCP'})

        return pd.DataFrame(coverage_data)

    def plot_results_from_saved(self, save_path: str = None) -> None:
        """Create and display the results visualization from saved results (without regenerating visualization data)."""
        if self.avg_miscov_vanilla is None or self.avg_miscov_gcfcp is None:
            raise ValueError("Must run Monte Carlo simulations or load results first")

        # Create coverage DataFrame
        coverage_df = self.create_coverage_dataframe()

        # Set up the plot (simplified version without threshold plots since we don't have test data)
        cp = sns.color_palette()
        sns.set(font="DejaVu Sans")
        sns.set_style("whitegrid", {'axes.grid': False})
        fig = plt.figure(figsize=(8, 6))

        # Coverage plot
        ax = fig.add_subplot(1, 1, 1)
        sns.barplot(
            data=coverage_df,
            x='Range',
            y='Miscoverage',
            hue='Method',
            palette=cp,
            ax=ax
        )
        ax.axhline(self.alpha, color='red', linestyle='--', linewidth=2, label=f'Target α={self.alpha}')
        ax.set_ylabel("Miscoverage", fontsize=18, labelpad=10)
        ax.set_xlabel("Groups", fontsize=18, labelpad=10)
        ax.set_ylim(0., 0.2)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.legend(fontsize=16, loc='upper right')
        ax.set_title(f"Coverage Results (α={self.alpha})", fontsize=18, pad=12)

        plt.tight_layout(pad=3)

        if save_path:
            plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
            plt.savefig(f'{save_path}.pdf', bbox_inches='tight')

        # plt.show()

    def plot_results(self, save_path: str = None) -> None:
        """Create and display the results visualization."""
        if self.avg_miscov_vanilla is None or self.avg_miscov_gcfcp is None:
            raise ValueError("Must run Monte Carlo simulations first")

        # Check if loaded stored data, only plot the bar chart.
        # if self.loaded:
        #     self.plot_results_from_saved(save_path)
        #     return

        # Generate visualization data
        viz_data = self.generate_visualization_data()

        # Sort data for plotting
        sort_order = np.argsort(viz_data['X_test'])
        X_test_s = viz_data['X_test'][sort_order]
        S_test_s = viz_data['S_test'][sort_order]
        taus_vanilla_s = viz_data['taus_vanilla'][sort_order]
        taus_centralized_gcfcp_s = viz_data['taus_centralized_gcfcp'][sort_order] if self.run_centralized_gcfcp else None
        taus_gcfcp_s = viz_data['taus_gcfcp'][sort_order]

        # Create coverage DataFrame
        coverage_df = self.create_coverage_dataframe()

        # Set up the plot
        cp = sns.color_palette()
        sns.set(font="DejaVu Sans")
        sns.set_style("whitegrid", {'axes.grid': False})
        fig = plt.figure(figsize=(17.5, 6))

        # Centralized CP plot
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.plot(X_test_s, S_test_s, '.', alpha=0.2)
        ax1.plot(X_test_s, taus_vanilla_s, color=cp[0], lw=2, label='Centralized CP threshold')
        # ax1.set_ylim(0.9, 1)
        ax1.tick_params(axis='both', which='major', labelsize=14)
        ax1.set_xlabel("Predicted Class", fontsize=16, labelpad=10)
        ax1.set_ylabel("Score", fontsize=16, labelpad=10)
        ax1.set_title("Centralized CP", fontsize=18, pad=12)
        for g in self.groups:
            ax1.axvspan(g[0], g[1], facecolor='grey', alpha=0.25)

        # CondDCP & GC-FCP plot
        ax2 = fig.add_subplot(1, 3, 2, sharex=ax1, sharey=ax1)
        ax2.plot(X_test_s, S_test_s, '.', alpha=0.2)
        ax2.plot(X_test_s, taus_gcfcp_s, color=cp[1], lw=2, linestyle='--', label='GC-FCP threshold')
        if self.run_centralized_gcfcp and taus_centralized_gcfcp_s is not None:
            ax2.plot(X_test_s, taus_centralized_gcfcp_s, color=cp[2], lw=2, label='CondDCP threshold')
        ax2.tick_params(axis='both', which='major', direction='out', labelsize=14)
        ax2.set_xlabel("Predicted Class", fontsize=16, labelpad=10)
        ax2.set_ylabel("Score", fontsize=16, labelpad=10)
        ax2.set_title("CondDCP & GC-FCP" if self.run_centralized_gcfcp else "GC-FCP", fontsize=18, pad=12)
        # ax2.set_ylim(0.9, 1)
        for g in self.groups:
            ax2.axvspan(g[0], g[1], facecolor='grey', alpha=0.25)
        ax2.legend(fontsize=12, loc='upper right')

        # Coverage plot
        ax3 = fig.add_subplot(1, 3, 3)
        sns.barplot(
            data=coverage_df,
            x='Range',
            y='Miscoverage',
            hue='Method',
            palette=cp,
            ax=ax3
        )
        ax3.axhline(self.alpha, color='red')
        ax3.set_ylabel("Miscoverage", fontsize=18, labelpad=10)
        ax3.set_xlabel("Groups", fontsize=18, labelpad=10)
        ax3.set_ylim(0., 0.2)
        ax3.tick_params(axis='both', which='major', labelsize=14)
        ax3.legend(fontsize=16, loc='upper right')

        plt.tight_layout(pad=3)

        if save_path:
            plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
            plt.savefig(f'{save_path}.pdf', bbox_inches='tight')

        # plt.show()

    def print_results(self) -> None:
        """Print summary of results."""
        if self.avg_miscov_vanilla is None or self.avg_miscov_gcfcp is None:
            raise ValueError("Must run Monte Carlo simulations or load results first")

        print("\n" + "=" * 80)
        print(f"CONFORMAL PREDICTION EXPERIMENT RESULTS ({self.dataset} WITH GC-FCP)")
        print("=" * 80)
        print(f"Number of Monte Carlo simulations: {self.num_mc}")
        print(f"Significance level (α): {self.alpha}")
        print(f"Target coverage: {1 - self.alpha:.1%}")
        print(f"Number of clients: {self.K}")
        print(f"GC-FCP compression: {self.compression}")
        print(f"Run centralized_gcfcp: {self.run_centralized_gcfcp}")

        print("\nMISCOVERAGE BY EVALUATION RANGE:")
        print("-" * 60)
        for i, range_name in enumerate(self.eval_ranges):
            vanilla_miscov = self.avg_miscov_vanilla[i + 1]
            centralized_gcfcp_miscov = self.avg_miscov_centralized_gcfcp[i + 1] if (
                        self.run_centralized_gcfcp and self.avg_miscov_centralized_gcfcp is not None) else "N/A"
            gs_miscov = self.avg_miscov_gcfcp[i + 1]
            print(
                f"{range_name:12} | Centralized CP: {vanilla_miscov:.3f} | CondDCP: {centralized_gcfcp_miscov} | GC-FCP: {gs_miscov:.3f}")

        print("\nMARGINAL MISCOVERAGE:")
        print("-" * 40)
        print(f"Centralized CP: {self.avg_miscov_vanilla[0]:.3f}")
        if self.run_centralized_gcfcp and self.avg_miscov_centralized_gcfcp is not None:
            print(f"CondDCP:   {self.avg_miscov_centralized_gcfcp[0]:.3f}")
        print(f"GC-FCP: {self.avg_miscov_gcfcp[0]:.3f}")

        print("\nSTANDARD ERRORS:")
        print("-" * 40)
        if self.miscov_vanilla_mc is not None:
            vanilla_se = np.std(self.miscov_vanilla_mc, axis=0) / np.sqrt(self.num_mc)
            centralized_gcfcp_se = np.std(self.miscov_centralized_gcfcp_mc, axis=0) / np.sqrt(self.num_mc) if (
                        self.run_centralized_gcfcp and self.miscov_centralized_gcfcp_mc is not None) else None
            gs_se = np.std(self.miscov_gcfcp_mc, axis=0) / np.sqrt(self.num_mc)

            for i, range_name in enumerate(self.eval_ranges):
                v_se = vanilla_se[i + 1]
                n_se = centralized_gcfcp_se[i + 1] if centralized_gcfcp_se is not None else "N/A"
                g_se = gs_se[i + 1]
                print(f"{range_name:12} | Centralized CP: {v_se:.4f} | CondDCP: {n_se} | GC-FCP: {g_se:.4f}")

        print("\nAVERAGE SET SIZES BY EVALUATION RANGE:")
        print("-" * 60)
        if self.avg_set_size_vanilla is not None:
            for i, range_name in enumerate(self.eval_ranges):
                vanilla_size = self.avg_set_size_vanilla[i + 1]
                centralized_gcfcp_size = self.avg_set_size_centralized_gcfcp[i + 1] if (
                            self.run_centralized_gcfcp and self.avg_set_size_centralized_gcfcp is not None) else "N/A"
                gs_size = self.avg_set_size_gcfcp[i + 1]
                print(
                    f"{range_name:12} | Centralized CP: {vanilla_size:.3f} | CondDCP: {centralized_gcfcp_size} | GC-FCP: {gs_size:.3f}")

        print("\nMARGINAL AVERAGE SET SIZE:")
        print("-" * 40)
        if self.avg_set_size_vanilla is not None:
            print(f"Centralized CP: {self.avg_set_size_vanilla[0]:.3f}")
            if self.run_centralized_gcfcp and self.avg_set_size_centralized_gcfcp is not None:
                print(f"CondDCP:   {self.avg_set_size_centralized_gcfcp[0]:.3f}")
            print(f"GC-FCP: {self.avg_set_size_gcfcp[0]:.3f}")


class AlphaExperiments:
    def __init__(self, alphas: List[float], **kwargs):
        self.alphas = sorted(alphas)
        self.kwargs = kwargs
        self.experiments = {}
        self.results = {}

        self.methods = ['Centralized CP', 'central', 'GC-FCP']
        self.colors = {'Centralized CP': 'darkblue', 'central': 'darkgreen', 'GC-FCP': 'darkred'}
        self.linestyles = {'Centralized CP': '-', 'central': '-.', 'GC-FCP': '-'}
        self.markerstyles = {'Centralized CP': '*', 'central': '^', 'GC-FCP': 'o'}

        if self.kwargs.get('dataset') == 'cifar10':
            if self.kwargs.get('overlap'):
                self.groups = [[0, 4], [2, 6], [4, 8], [6, 10]]
                self.eval_ranges = ['{0,1,2,3}', '{2,3,4,5}', '{4,5,6,7}', '{6,7,8,9}']
            else:
                self.groups = [[0, 2], [2, 4], [4, 6], [6, 8], [8, 10]]
                self.eval_ranges = ['{0,1}', '{2,3}', '{4,5}', '{6,7}', '{8,9}']
        elif self.kwargs.get('dataset') == 'pathmnist':
            if self.kwargs.get('overlap'):
                self.groups = [[0, 3], [1, 4], [2, 5], [3, 6], [4, 9]]
                self.eval_ranges = ['{0,1,2}', '{1,2,3}', '{2,3,4}', '{3,4,5}', '{4,5,6,7,8}']
            else:
                # optional disjoint fallback
                self.groups = [[0, 2], [2, 4], [4, 6], [6, 8], [8, 9]]
                self.eval_ranges = ['{0,1}', '{2,3}', '{4,5}', '{6,7}', '{8}']
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")


    def run(self):
        for alpha in self.alphas:
            print(f"\nRunning experiment for alpha = {alpha}")
            exp = Experiment(alpha=alpha, **self.kwargs)
            exp.run_monte_carlo()
            self.experiments[alpha] = exp
            self.results[alpha] = {
                'miscov': {
                    'Centralized CP': exp.avg_miscov_vanilla,
                    'central': exp.avg_miscov_centralized_gcfcp if exp.run_centralized_gcfcp else None,
                    'GC-FCP': exp.avg_miscov_gcfcp
                },
                'set_size': {
                    'Centralized CP': exp.avg_set_size_vanilla,
                    'central': exp.avg_set_size_centralized_gcfcp if exp.run_centralized_gcfcp else None,
                    'GC-FCP': exp.avg_set_size_gcfcp
                },
                'std_miscov': {
                    'Centralized CP': np.std(exp.miscov_vanilla_mc, axis=0) / np.sqrt(exp.num_mc),
                    'central': np.std(exp.miscov_centralized_gcfcp_mc, axis=0) / np.sqrt(exp.num_mc) if exp.run_centralized_gcfcp else None,
                    'GC-FCP': np.std(exp.miscov_gcfcp_mc, axis=0) / np.sqrt(exp.num_mc)
                },
                'std_set_size': {
                    'Centralized CP': np.std(exp.set_size_vanilla_mc, axis=0) / np.sqrt(exp.num_mc),
                    'central': np.std(exp.set_size_centralized_gcfcp_mc, axis=0) / np.sqrt(exp.num_mc) if exp.run_centralized_gcfcp else None,
                    'GC-FCP': np.std(exp.set_size_gcfcp_mc, axis=0) / np.sqrt(exp.num_mc)
                }
            }

    def save_results(self, filename: str) -> None:
        """Save all alpha experiment results to file."""
        results_data = {
            'alphas': self.alphas,
            'kwargs': self.kwargs,
            'results': {},
            'timestamp': datetime.now().isoformat()
        }

        # Convert numpy arrays to lists for JSON serialization
        for alpha in self.alphas:
            if alpha in self.results:
                alpha_results = {}
                for key, methods_dict in self.results[alpha].items():
                    alpha_results[key] = {}
                    for method, values in methods_dict.items():
                        if values is not None:
                            alpha_results[key][method] = values.tolist() if isinstance(values, np.ndarray) else values
                        else:
                            alpha_results[key][method] = None
                results_data['results'][str(alpha)] = alpha_results

        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"Alpha experiment results saved to {filename}")

    def load_results(self, filename: str) -> None:
        """Load alpha experiment results from file."""
        with open(filename, 'r') as f:
            data = json.load(f)

        self.alphas = data['alphas']
        self.kwargs = data['kwargs']

        # Reconstruct results
        self.results = {}
        for alpha_str, alpha_results in data['results'].items():
            alpha = float(alpha_str)
            self.results[alpha] = {}
            for key, methods_dict in alpha_results.items():
                self.results[alpha][key] = {}
                for method, values in methods_dict.items():
                    if values is not None:
                        self.results[alpha][key][method] = np.array(values)
                    else:
                        self.results[alpha][key][method] = None

        print(f"Alpha experiment results loaded from {filename}")

    def plot_coverage(self, save_path: str = None):
        if not self.results:
            raise ValueError("Must run experiments or load results first")

        # Get experiment parameters from first alpha
        first_alpha = self.alphas[0]
        if first_alpha in self.results:
            # Infer parameters from results structure
            num_groups = len(self.results[first_alpha]['miscov']['Centralized CP']) - 1
            eval_ranges = self.eval_ranges[:num_groups]
        else:
            raise ValueError("No results found for plotting")

        num_plots = 1 + num_groups  # marginal + groups
        labels = ['Marginal'] + eval_ranges

        methods = self.methods
        colors = self.colors
        linestyles = self.linestyles
        markerstyles = self.markerstyles

        fig, axs = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))

        x = [1 - a for a in self.alphas]
        for i in range(num_plots):
            ax = axs[i] if num_plots > 1 else axs
            ax.plot(x, x, '--', color='black', linewidth=2.5)
            for method in methods:
                # Check if method exists in results
                if method not in self.results[self.alphas[0]]['miscov'] or self.results[self.alphas[0]]['miscov'][method] is None:
                    continue

                mean_cov = [1 - self.results[a]['miscov'][method][i] for a in self.alphas]
                std_cov = [self.results[a]['std_miscov'][method][i] for a in self.alphas] if \
                self.results[self.alphas[0]]['std_miscov'][method] is not None else None
                ax.plot(x, mean_cov, label=method, color=colors[method], linestyle=linestyles[method],
                        marker=markerstyles[method], linewidth=2.5, markersize=8)
                if std_cov is not None:
                    lower = np.array(mean_cov) - 1.96 * np.array(std_cov)
                    upper = np.array(mean_cov) + 1.96 * np.array(std_cov)
                    ax.fill_between(x, lower, upper, color=colors[method], alpha=0.3)

            ax.set_title(labels[i])
            ax.set_xlabel(r'$1 - \alpha$')
            ax.set_ylabel('Coverage')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(True)
            if i == num_plots - 1:
                ax.legend()

            plt.tight_layout()

            if save_path:
                plt.savefig(f'{save_path}_coverage.pdf', bbox_inches='tight')
            plt.savefig(f'{save_path}_coverage.png', dpi=300, bbox_inches='tight')

            # plt.show()

    def plot_set_size(self, normalize: bool = True, save_path: str = None):
        if not self.results:
            raise ValueError("Must run experiments or load results first")

        # Get experiment parameters from first alpha
        first_alpha = self.alphas[0]
        if first_alpha in self.results:
            num_groups = len(self.results[first_alpha]['set_size']['Centralized CP']) - 1
            eval_ranges = self.eval_ranges[:num_groups]
        else:
            raise ValueError("No results found for plotting")

        num_plots = 1 + num_groups  # marginal + groups
        labels = ['Marginal'] + eval_ranges
        methods = self.methods
        colors = self.colors
        linestyles = self.linestyles
        markerstyles = self.markerstyles
        div = 10 if normalize else 1

        fig, axs = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))

        for i in range(num_plots):
            ax = axs[i] if num_plots > 1 else axs
            for method in methods:
                # Check if method exists in results
                if method not in self.results[self.alphas[0]]['set_size'] or self.results[self.alphas[0]]['set_size'][method] is None:
                    continue

                mean_size = [self.results[a]['set_size'][method][i] / div for a in self.alphas]
                std_size = [self.results[a]['std_set_size'][method][i] / div for a in self.alphas] if \
                self.results[self.alphas[0]]['std_set_size'][method] is not None else None
                x = [1 - a for a in self.alphas]
                ax.plot(x, mean_size, label=method, color=colors[method], linestyle=linestyles[method],
                        marker=markerstyles[method], linewidth=2.5, markersize=8)
                if std_size is not None:
                    lower = np.array(mean_size) - 1.96 * np.array(std_size)
                    upper = np.array(mean_size) + 1.96 * np.array(std_size)
                    ax.fill_between(x, lower, upper, color=colors[method], alpha=0.3)

            ax.set_title(labels[i])
            ax.set_xlabel(r'$1 - \alpha$')
            ax.set_ylabel('Normalized Set Size' if normalize else 'Set Size')
            ax.set_xlim(0, 1)
            ax.grid(True)
            if i == num_plots - 1:
                ax.legend()

            plt.tight_layout()

            if save_path:
                plt.savefig(f'{save_path}_set_size.pdf', bbox_inches='tight')
            plt.savefig(f'{save_path}_set_size.png', dpi=300, bbox_inches='tight')

            # plt.show()


def main_illustrate(args):
    """Main execution function for single alpha illustration."""
    alpha = 0.1

    # Generate filename based on parameters
    overlap_str = "overlap" if args.overlap else "disjoint"
    filename = f"results/{args.dataset}_illustration_K{args.clients}_mc{args.times}_delta{args.delta}_{overlap_str}_alpha{alpha}.json"

    if args.load_path:
        experiment = Experiment(K=args.clients, alpha=alpha, run_centralized_gcfcp=False)
        experiment.load_results(args.load_path)
    else:
        experiment = Experiment(
            K=args.clients,
            alpha=alpha,
            num_mc=args.times,
            n_jobs=10,
            delta=args.delta,
            overlap=args.overlap,
            run_centralized_gcfcp=False,
            dataset=args.dataset,
            model_path=args.model_path
        )

        # Run Monte Carlo simulations
        experiment.run_monte_carlo()

        # Save results
        experiment.save_results(filename)

    # Print results
    experiment.print_results()

    # Create visualization
    if args.overlap:
        experiment.plot_results(save_path=f'figures/{args.dataset}_dual_gcfcp')
    else:
        experiment.plot_results(save_path=f'figures/{args.dataset}_dual_gcfcp_disjoint')


def main_alpha(args):
    """Main execution function for alpha experiments."""
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # Generate filename based on parameters
    overlap_str = "overlap" if args.overlap else "disjoint"
    filename = f"results/{args.dataset}_alpha_K{args.clients}_mc{args.times}_delta{args.delta}_{overlap_str}.json"

    if args.load_path:
        # Load results from specified path
        experiment = AlphaExperiments(alphas=alphas, **args.kwargs)
        experiment.load_results(args.load_path)
    else:
        # Run new experiments
        experiment = AlphaExperiments(
            alphas=alphas,
            K=args.clients,
            num_mc=args.times,
            n_jobs=10,
            delta=args.delta,
            overlap=args.overlap,
            run_centralized_gcfcp=False,
            dataset=args.dataset,
            model_path=args.model_path
        )

        # Run experiments for varying alphas
        experiment.run()

        # Save results
        experiment.save_results(filename)

    # Create visualizations
    if args.overlap:
        experiment.plot_coverage(save_path=f'figures/{args.dataset}_dual_gcfcp')
        experiment.plot_set_size(save_path=f'figures/{args.dataset}_dual_gcfcp')
    else:
        experiment.plot_coverage(save_path=f'figures/{args.dataset}_dual_gcfcp_disjoint')
        experiment.plot_set_size(save_path=f'figures/{args.dataset}_dual_gcfcp_disjoint')


def main_plot_only(args):
    """Main function for plotting only from saved results."""
    if not args.load_path:
        raise ValueError("load_path must be specified for plot_only mode")

    if not os.path.exists(args.load_path):
        raise FileNotFoundError(f"Results file not found: {args.load_path}")

    # Determine if it's alpha experiment or single experiment based on filename or content
    with open(args.load_path, 'r') as f:
        data = json.load(f)

    if 'alphas' in data:
        # Alpha experiment
        experiment = AlphaExperiments(alphas=data['alphas'], overlap=args.overlap, dataset=args.dataset)
        experiment.load_results(args.load_path)

        # Create visualizations
        if experiment.kwargs.get('overlap', False):
            experiment.plot_coverage(save_path=f'figures/{args.dataset}_dual_gcfcp')
            experiment.plot_set_size(save_path=f'figures/{args.dataset}_dual_gcfcp')
        else:
            experiment.plot_coverage(save_path=f'figures/{args.dataset}_dual_gcfcp_disjoint')
            experiment.plot_set_size(save_path=f'figures/{args.dataset}_dual_gcfcp_disjoint')
    else:
        # Single alpha experiment
        experiment = Experiment()
        experiment.load_results(args.load_path)
        experiment.print_results()

        # Create visualization
        if experiment.overlap:
            experiment.plot_results(save_path=f'figures/{args.dataset}_dual_gcfcp')
        else:
            experiment.plot_results(save_path=f'figures/{args.dataset}_dual_gcfcp_disjoint')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment.")
    parser.add_argument('--exp', type=str, default='plot_only',
                        choices=['illustration', 'alpha', 'plot_only'],
                        help='Choose the experiment you want to run')
    parser.add_argument('--times', type=int, default=10,
                        help='Number of times to run the experiment')
    parser.add_argument('--delta', type=float, default=0.01,
                        help='T-Digest Parameter')
    parser.add_argument('--clients', type=int, default=5,
                        help='Number of clients')
    parser.add_argument('--overlap', action='store_true', default=False,
                        help='Overlap flag')
    parser.add_argument('--load_path', type=str, default=None,
                        help='Path to saved results file for loading/plotting')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'pathmnist'],
                        help='Dataset for the experiment')
    parser.add_argument('--model_path', type=str, default='checkpoints/path_cnn.pt',
                        help='Path to MedMNIST CNN checkpoint (PathMNIST only)')

    args = parser.parse_args()

    # Create figures directory if it doesn't exist
    os.makedirs('figures', exist_ok=True)

    if args.exp == 'illustration':
        main_illustrate(args)
    elif args.exp == 'alpha':
        main_alpha(args)
    elif args.exp == 'plot_only':
        main_plot_only(args)
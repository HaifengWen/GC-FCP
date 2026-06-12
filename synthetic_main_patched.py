from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.stats import truncnorm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import linprog
import pandas as pd
from multiprocessing import Pool, cpu_count
from typing import Tuple, List, Dict, Any
import warnings
from tqdm import tqdm
import time
from tdigest import TDigest
import matplotlib as mpl


warnings.filterwarnings('ignore')

plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
font = {'family': 'Times New Roman',
        # 'weight': 'bold',
        'size': 16}

mpl.rc('font', **font)


class SyntheticDataDualExperiment:
    """
    A class for running conformal prediction experiments on synthetic data with overlapping groups.

    This experiment compares vanilla conformal prediction, a naive dual LP approach, and GC-FCP
    for conditional conformal prediction across overlapping groups.
    """

    def __init__(self,
                 K: int = 4,
                 label_shifts: bool = False,
                 n_k: np.ndarray = None,
                 pi_k: List[int] = None,
                 alpha: float = 0.1,
                 n_test: int = 2000,
                 bounds: Tuple[float, float] = (0, 5),
                 poly_degree: int = 4,
                 num_mc: int = 10,
                 n_jobs: int = -1,
                 delta: float = 0.01,      # delta factor for GC-FCP
                 compression: int = 25):  # compression factor for GC-FCP
        """
        Initialize the experiment parameters.

        Args:
            K: Number of groups
            n_k: Sample sizes for each group
            alpha: Significance level
            n_test: Number of test samples
            bounds: Bounds for truncated normal distribution
            poly_degree: Degree of polynomial features
            num_mc: Number of Monte Carlo simulations
            n_jobs: Number of parallel jobs (-1 for all CPUs)
            compression: Compression factor δ for GC-FCP (higher = more accurate, more points)
        """
        self.K = K
        self.label_shifts = label_shifts
        self.n_k = n_k if n_k is not None else np.array([1000, 333, 333, 333])
        self.pi_k = pi_k if pi_k is not None else [1 / K] * K
        self.alpha = alpha
        self.n_test = n_test
        self.bounds = bounds
        self.poly_degree = poly_degree
        self.num_mc = num_mc
        self.n_jobs = n_jobs if n_jobs != -1 else cpu_count()
        self.compression = compression
        self.delta = delta

        # Derived parameters
        self.lambda_k = [self.pi_k[k] / (self.n_k[k] + 1) for k in range(K)]
        self.w_test = sum(self.lambda_k)
        self.poly = PolynomialFeatures(poly_degree)

        # Overlapping groups
        self.groups = [[0, 2], [1, 3], [2, 4], [3, 5]]
        self.d = len(self.groups)

        # Evaluation ranges (disjoint for miscoverage computation)
        self.eval_ranges = ['[0,2]', '[1,3]', '[2,4]', '[3,5]']

        # Results storage
        self.miscov_vanilla_mc = None
        self.miscov_naive_mc = None
        self.miscov_gs_mc = None  #
        self.avg_miscov_vanilla = None
        self.avg_miscov_naive = None
        self.avg_miscov_gs = None  #

        self.miscov_fedcp_mc = None
        self.miscov_condcp_mc = None
        self.avg_miscov_fedcp = None
        self.avg_miscov_condcp = None


    def generate_X(self, mu: float, sigma: float, size: int) -> np.ndarray:
        """Generate X from truncated normal distribution."""
        a = (self.bounds[0] - mu) / sigma
        b = (self.bounds[1] - mu) / sigma
        return truncnorm.rvs(a, b, loc=mu, scale=sigma, size=size)

    def generate_Y(self, X: np.ndarray, k: int) -> np.ndarray:
        """Generate Y given X and group k."""
        pois = np.random.poisson(np.sin(X) ** 2 + 0.1, size=len(X))
        eps1 = np.random.normal(0, 1, size=len(X))
        U = np.random.uniform(0, 1, size=len(X))
        eps2 = np.random.normal(0, 1, size=len(X))
        # normal = 0
        normal = np.random.normal(0, 0.01 * k ** 2, size=len(X)) if self.label_shifts else 0

        return pois + 0.03 * X * eps1 + 25 * (U < 0.01) * eps2 + normal

    def generate_training_data(self, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate training data for all groups."""
        if seed is not None:
            np.random.seed(seed)

        X_train, Y_train = [], []

        for k in range(1, self.K + 1):
            mu = 0.5 + 4 * (k - 1) / (self.K - 1)
            sigma = 0.5 + 0.1 * (k - 1)
            X_k = self.generate_X(mu, sigma, self.n_k[k - 1])
            Y_k = self.generate_Y(X_k, k)
            X_train.append(X_k)
            Y_train.append(Y_k)

        X_train_pooled = np.concatenate(X_train).reshape(-1, 1)
        Y_train_pooled = np.concatenate(Y_train)

        return X_train_pooled, Y_train_pooled

    def generate_calibration_data(self, reg: LinearRegression) -> Dict[str, Any]:
        """Generate calibration data and compute residuals."""
        X_calib, Y_calib, S_calib, w_calib = [], [], [], []

        for k in range(self.K):
            mu = 0.5 + 4 * k / (self.K - 1)
            sigma = 0.5 + 0.1 * k
            X_k = self.generate_X(mu, sigma, self.n_k[k])
            Y_k = self.generate_Y(X_k, k + 1)
            S_k = np.abs(Y_k - reg.predict(self.poly.fit_transform(X_k.reshape(-1, 1))))
            w_k = np.full(len(X_k), self.lambda_k[k])

            X_calib.append(X_k)
            Y_calib.append(Y_k)
            S_calib.append(S_k)
            w_calib.append(w_k)

        return {
            'X_calib': X_calib,
            'Y_calib': Y_calib,
            'S_calib': S_calib,
            'w_calib': w_calib
        }

    def compute_vanilla_quantiles(self, S_calib: List[np.ndarray]) -> Tuple[float, float]:
        """Compute vanilla conformal prediction quantiles."""
        S_calib_pooled = np.concatenate(S_calib)
        n = len(S_calib_pooled)

        q_high = np.quantile(S_calib_pooled, (1 - self.alpha) * (1 + 1 / n))
        q_low = -q_high  # Assuming symmetry for intervals

        return q_low, q_high



    def compute_fedcp_quantiles(self, S_calib: List[np.ndarray]) -> Tuple[float, float]:
        """Compute FedCP quantiles (Lu et al., 2023) under mixture weights π_k.

        This is a federated analogue of vanilla split CP: it uses a single global threshold based on
        all client calibration scores with the (n_k + 1) correction via the weights λ_k = π_k/(n_k + 1).
        """
        S_all = np.concatenate(S_calib)
        w_all = np.concatenate([np.full(len(S_calib[k]), self.lambda_k[k]) for k in range(self.K)])

        # Weighted (1 - alpha)-quantile of the finite scores; if the target mass exceeds the total finite mass,
        # the quantile is effectively +infty (due to the implicit "add-one" mass per client).
        target = 1 - self.alpha
        order = np.argsort(S_all)
        S_sorted = S_all[order]
        w_sorted = w_all[order]
        cum_w = np.cumsum(w_sorted)

        if cum_w[-1] < target:
            q_high = np.inf
        else:
            q_high = S_sorted[np.searchsorted(cum_w, target, side='left')]

        q_low = -q_high  # symmetric intervals
        return q_low, q_high

    @staticmethod
    def find_S_star(X_test_point: float, X_all: np.ndarray, S_all: np.ndarray,
                    w_all: np.ndarray, w_test: float, alpha: float, groups: List[List[int]]) -> float:
        """Find S* using binary search with dual LP solved via linprog (single eta_test variable)."""
        d = len(groups)
        Phi_test = np.array([1 if g[0] <= X_test_point < g[1] else 0 for g in groups])

        n = len(X_all)
        # Precompute Phi_all (n x d)
        Phi_all = np.zeros((n, d))
        for i in range(n):
            Phi_all[i] = [1 if g[0] <= X_all[i] < g[1] else 0 for g in groups]

        # For linprog: variables [eta_0, ..., eta_{n-1}, eta_test]
        # A_eq (d x (n+1)): [Phi_all.T | Phi_test]
        A_eq = np.hstack((Phi_all.T, Phi_test.reshape(-1, 1)))

        # Right-hand side for equality constraints
        b_eq = np.zeros(d)

        # Bounds: list of (lb, ub) for each variable
        bounds = [(-w_all[i] * alpha, w_all[i] * (1 - alpha)) for i in range(n)]
        bounds.append((-w_test * alpha, w_test * (1 - alpha)))

        low = 0.0
        high = 100.0
        threshold = w_test * (1 - alpha)
        # threshold = w_test * np.random.uniform(-alpha, 1 - alpha)
        for _ in range(50):  # Reduced iterations for precision
            mid = (low + high) / 2
            # Objective for linprog: minimize - (sum eta_i S_i + mid eta_test) => c = [-S_all, -mid]
            c = np.append(-S_all, -mid)

            # Solve LP
            res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

            if not res.success:
                return low  # Fallback

            eta_test_val = res.x[-1]

            # Check if eta_test saturates at upper bound
            if eta_test_val >= threshold:
                high = mid
            else:
                low = mid

            if high - low < 1e-6:
                break

        return high  # Return high for the minimal saturating threshold

    # Simulate federated GC-FCP: each client computes local TDigest per atom, server merges
    def compute_gs_pseudo_data_fed(self, X_calib: List[np.ndarray], S_calib: List[np.ndarray],
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
    # Find S* for GC-FCP using binary search on coreset
    def find_S_star_gs(X_test_point: float, pseudo_S: np.ndarray, pseudo_w: np.ndarray,
                       pseudo_Phi: np.ndarray, w_test: float, alpha: float, groups: List[List[int]]) -> float:
        """Find S* using binary search with dual LP on GC-FCP coreset."""
        d = len(groups)
        Phi_test = np.array([1 if g[0] <= X_test_point < g[1] else 0 for g in groups])

        m = len(pseudo_S)
        # A_eq (d x (m+1)): [pseudo_Phi.T | Phi_test]
        A_eq = np.hstack((pseudo_Phi.T, Phi_test.reshape(-1, 1)))

        # Right-hand side for equality constraints
        b_eq = np.zeros(d)

        # Bounds: list of (lb, ub) for each pseudo-point
        bounds = [(-pseudo_w[i] * alpha, pseudo_w[i] * (1 - alpha)) for i in range(m)]
        bounds.append((-w_test * alpha, w_test * (1 - alpha)))

        low = 0.0
        high = 100.0
        threshold = w_test * (1 - alpha)
        # threshold = w_test * np.random.uniform(-alpha, 1 - alpha)
        for _ in range(50):
            mid = (low + high) / 2
            # Objective: c = [-pseudo_S, -mid]
            c = np.append(-pseudo_S, -mid)

            # Solve LP
            res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

            if not res.success:
                return low  # Fallback

            eta_test_val = res.x[-1]

            if eta_test_val >= threshold:
                high = mid
            else:
                low = mid

            if high - low < 1e-6:
                break

        return high

    def generate_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate test data from mixture distribution."""
        X_test, Y_test = [], []

        for _ in range(self.n_test):
            k_idx = np.random.choice(range(self.K), p=self.pi_k)
            mu = 0.5 + 4 * k_idx / (self.K - 1)
            sigma = 0.5 + 0.1 * k_idx
            X = self.generate_X(mu, sigma, 1)[0]
            Y = self.generate_Y(np.array([X]), k_idx + 1)[0]
            X_test.append(X)
            Y_test.append(Y)

        return np.array(X_test), np.array(Y_test)

    def compute_prediction_intervals(self, X_test: np.ndarray, Y_test_hat: np.ndarray,
                                     X_all: np.ndarray, S_all: np.ndarray,
                                     w_all: np.ndarray,
                                     pseudo_S: np.ndarray, pseudo_w: np.ndarray,
                                     pseudo_Phi: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Compute prediction intervals for naive dual and GC-FCP."""
        n_test = len(X_test)

        # Naive (exact)
        # start_time = time.time()
        # taus_naive = np.zeros(n_test)
        # for i in range(n_test):
        #     taus_naive[i] = self.find_S_star(X_test[i], X_all, S_all, w_all, self.w_test)
        # end_time = time.time()
        # print(f'Centralized time (per test point): {(end_time - start_time)/n_test}')

        start_time = time.time()
        with Pool(self.n_jobs) as pool:
            args = [(X_test[i], X_all, S_all, w_all, self.w_test, self.alpha, self.groups) for i in range(n_test)]
            taus_naive = np.array(pool.starmap(SyntheticDataDualExperiment.find_S_star, args))
        end_time = time.time()

        lbs_naive = Y_test_hat - taus_naive
        ubs_naive = Y_test_hat + taus_naive



        # CondCP (centralized): special case of GC-FCP with a single population (uniform weights)
        n = len(X_all)
        w_all_condcp = np.full(n, 1.0 / (n + 1))
        w_test_condcp = 1.0 / (n + 1)

        start_time = time.time()
        with Pool(self.n_jobs) as pool:
            args = [(X_test[i], X_all, S_all, w_all_condcp, w_test_condcp, self.alpha, self.groups) for i in range(n_test)]
            taus_condcp = np.array(pool.starmap(SyntheticDataDualExperiment.find_S_star, args))
        end_time = time.time()

        lbs_condcp = Y_test_hat - taus_condcp
        ubs_condcp = Y_test_hat + taus_condcp

        # GC-FCP
        # start_time = time.time()
        # taus_gs = np.zeros(n_test)
        # for i in range(n_test):
        #     taus_gs[i] = self.find_S_star_gs(X_test[i], pseudo_S, pseudo_w, pseudo_Phi, self.w_test)
        #
        # end_time = time.time()
        # print(f'GC-FCP time (per test point): {(end_time - start_time)/n_test}')

        start_time = time.time()
        with Pool(self.n_jobs) as pool:
            args = [(X_test[i], pseudo_S, pseudo_w, pseudo_Phi, self.w_test, self.alpha, self.groups) for i in
                    range(n_test)]
            taus_gs = np.array(pool.starmap(SyntheticDataDualExperiment.find_S_star_gs, args))
        end_time = time.time()

        lbs_gs = Y_test_hat - taus_gs
        ubs_gs = Y_test_hat + taus_gs

        return {
            'naive': (lbs_naive, ubs_naive),
            'condcp': (lbs_condcp, ubs_condcp),
            'gs': (lbs_gs, ubs_gs)
        }

    def run_single_simulation(self, mc_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run a single Monte Carlo simulation."""
        np.random.seed(mc_idx)

        # Generate training data and fit model
        X_train, Y_train = self.generate_training_data()
        reg = LinearRegression().fit(self.poly.fit_transform(X_train), Y_train)

        # Generate calibration data
        calib_data = self.generate_calibration_data(reg)

        # Pool calibration data for naive
        X_all = np.concatenate(calib_data['X_calib'])
        S_all = np.concatenate(calib_data['S_calib'])
        w_all = np.concatenate(calib_data['w_calib'])

        # Compute GC-FCP pseudo-data in federated manner ()
        pseudo_S, pseudo_w, pseudo_Phi = self.compute_gs_pseudo_data_fed(calib_data['X_calib'], calib_data['S_calib'], calib_data['w_calib'])

        # Compute vanilla quantiles
        q_low_vanilla, q_high_vanilla = self.compute_vanilla_quantiles(calib_data['S_calib'])


        # Compute FedCP quantiles
        q_low_fedcp, q_high_fedcp = self.compute_fedcp_quantiles(calib_data['S_calib'])
        # Generate test data
        X_test, Y_test = self.generate_test_data()
        Y_test_hat = reg.predict(self.poly.fit_transform(X_test.reshape(-1, 1)))

        # Compute intervals
        lbs_vanilla = Y_test_hat + q_low_vanilla
        ubs_vanilla = Y_test_hat + q_high_vanilla



        lbs_fedcp = Y_test_hat + q_low_fedcp
        ubs_fedcp = Y_test_hat + q_high_fedcp
        intervals = self.compute_prediction_intervals(X_test, Y_test_hat, X_all, S_all, w_all, pseudo_S, pseudo_w, pseudo_Phi)
        lbs_naive, ubs_naive = intervals['naive']
        lbs_condcp, ubs_condcp = intervals['condcp']
        lbs_gs, ubs_gs = intervals['gs']  #

        # Compute miscoverage: marginal + 4 ranges (len(eval_ranges))
        miscov_vanilla = np.zeros(5)  # Adjusted to 1 marginal + 4 groups
        miscov_fedcp = np.zeros(5)
        miscov_condcp = np.zeros(5)
        miscov_naive = np.zeros(5)
        miscov_gs = np.zeros(5)  #
        # Marginal
        miscov_vanilla[0] = np.mean((Y_test < lbs_vanilla) | (Y_test > ubs_vanilla))
        miscov_fedcp[0] = np.mean((Y_test < lbs_fedcp) | (Y_test > ubs_fedcp))
        miscov_condcp[0] = np.mean((Y_test < lbs_condcp) | (Y_test > ubs_condcp))
        miscov_naive[0] = np.mean((Y_test < lbs_naive) | (Y_test > ubs_naive))
        miscov_gs[0] = np.mean((Y_test < lbs_gs) | (Y_test > ubs_gs))  #

        # Overlapping groups
        for g in range(len(self.groups)):
            low_b, up_b = self.groups[g]
            mask = (X_test >= low_b) & (X_test < up_b)
            if np.sum(mask) > 0:
                miscov_vanilla[g + 1] = np.mean(
                    (Y_test[mask] < lbs_vanilla[mask]) | (Y_test[mask] > ubs_vanilla[mask])
                )
                miscov_fedcp[g + 1] = np.mean(
                    (Y_test[mask] < lbs_fedcp[mask]) | (Y_test[mask] > ubs_fedcp[mask])
                )
                miscov_condcp[g + 1] = np.mean(
                    (Y_test[mask] < lbs_condcp[mask]) | (Y_test[mask] > ubs_condcp[mask])
                )
                miscov_naive[g + 1] = np.mean(
                    (Y_test[mask] < lbs_naive[mask]) | (Y_test[mask] > ubs_naive[mask])
                )
                miscov_gs[g + 1] = np.mean(  #
                    (Y_test[mask] < lbs_gs[mask]) | (Y_test[mask] > ubs_gs[mask])
                )

        return miscov_vanilla, miscov_fedcp, miscov_condcp, miscov_naive, miscov_gs  # Updated

    def run_monte_carlo(self) -> None:
        """Run Monte Carlo simulations in parallel."""
        print(f"Running {self.num_mc} Monte Carlo simulations on {self.n_jobs} cores...")

        # with Pool(self.n_jobs) as pool:
        #     results = pool.map(self.run_single_simulation, tqdm(range(self.num_mc)))

        results = []
        for mc_idx in tqdm(range(self.num_mc)):
            results.append(self.run_single_simulation(mc_idx))

        # Unpack results
        miscov_vanilla_list, miscov_fedcp_list, miscov_condcp_list, miscov_naive_list, miscov_gs_list = zip(*results)  # Updated

        self.miscov_vanilla_mc = np.array(miscov_vanilla_list)
        self.miscov_fedcp_mc = np.array(miscov_fedcp_list)
        self.miscov_condcp_mc = np.array(miscov_condcp_list)
        self.miscov_naive_mc = np.array(miscov_naive_list)
        self.miscov_gs_mc = np.array(miscov_gs_list)  #

        # Compute averages
        self.avg_miscov_vanilla = np.mean(self.miscov_vanilla_mc, axis=0)
        self.avg_miscov_fedcp = np.mean(self.miscov_fedcp_mc, axis=0)
        self.avg_miscov_condcp = np.mean(self.miscov_condcp_mc, axis=0)
        self.avg_miscov_naive = np.mean(self.miscov_naive_mc, axis=0)
        self.avg_miscov_gs = np.mean(self.miscov_gs_mc, axis=0)  #

        print("Monte Carlo simulations completed.")

    def generate_visualization_data(self, seed: int = 42) -> Dict[str, Any]:
        """Generate data for visualization with fixed seed."""
        np.random.seed(seed)

        # Generate training data and fit model
        X_train, Y_train = self.generate_training_data()
        reg = LinearRegression().fit(self.poly.fit_transform(X_train), Y_train)

        # Generate calibration data
        calib_data = self.generate_calibration_data(reg)

        # Pool calibration data
        X_all = np.concatenate(calib_data['X_calib'])
        S_all = np.concatenate(calib_data['S_calib'])
        w_all = np.concatenate(calib_data['w_calib'])

        # Compute GS pseudo-data in federated manner ()
        pseudo_S, pseudo_w, pseudo_Phi = self.compute_gs_pseudo_data_fed(calib_data['X_calib'], calib_data['S_calib'], calib_data['w_calib'])

        # Compute vanilla quantiles
        q_low_vanilla, q_high_vanilla = self.compute_vanilla_quantiles(calib_data['S_calib'])

        # Compute FedCP quantiles
        q_low_fedcp, q_high_fedcp = self.compute_fedcp_quantiles(calib_data['S_calib'])

        # Generate test data
        X_test, Y_test = self.generate_test_data()
        Y_test_hat = reg.predict(self.poly.fit_transform(X_test.reshape(-1, 1)))

        # Compute intervals
        lbs_vanilla = Y_test_hat + q_low_vanilla
        ubs_vanilla = Y_test_hat + q_high_vanilla

        lbs_fedcp = Y_test_hat + q_low_fedcp
        ubs_fedcp = Y_test_hat + q_high_fedcp

        intervals = self.compute_prediction_intervals(X_test, Y_test_hat, X_all, S_all, w_all, pseudo_S, pseudo_w, pseudo_Phi)
        lbs_naive, ubs_naive = intervals['naive']
        lbs_condcp, ubs_condcp = intervals['condcp']
        lbs_gs, ubs_gs = intervals['gs']  #

        return {
            'X_test': X_test,
            'Y_test': Y_test,
            'Y_test_hat': Y_test_hat,
            'lbs_vanilla': lbs_vanilla,
            'ubs_vanilla': ubs_vanilla,
            'lbs_fedcp': lbs_fedcp,
            'ubs_fedcp': ubs_fedcp,
            'lbs_naive': lbs_naive,
            'ubs_naive': ubs_naive,
            'lbs_condcp': lbs_condcp,
            'ubs_condcp': ubs_condcp,
            'lbs_gs': lbs_gs,  #
            'ubs_gs': ubs_gs   #
        }

    def create_coverage_dataframe(self) -> pd.DataFrame:
        """Create DataFrame for coverage visualization."""
        if (self.avg_miscov_vanilla is None or self.avg_miscov_fedcp is None or self.avg_miscov_condcp is None
                or self.avg_miscov_naive is None or self.avg_miscov_gs is None):
            raise ValueError("Must run Monte Carlo simulations first")

        coverage_data = [
            {'Range': 'Marginal', 'Miscoverage': self.avg_miscov_vanilla[0], 'Method': 'Centralized CP'},
            {'Range': 'Marginal', 'Miscoverage': self.avg_miscov_condcp[0], 'Method': 'Centralized CondCP'},
            {'Range': 'Marginal', 'Miscoverage': self.avg_miscov_naive[0], 'Method': 'Centralized GC-FCP'},
            {'Range': 'Marginal', 'Miscoverage': self.avg_miscov_fedcp[0], 'Method': 'FedCP'},
            {'Range': 'Marginal', 'Miscoverage': self.avg_miscov_gs[0], 'Method': 'GC-FCP'},
        ]


        for r in range(len(self.eval_ranges)):
            coverage_data.extend([
                {'Range': self.eval_ranges[r], 'Miscoverage': self.avg_miscov_vanilla[r + 1], 'Method': 'Centralized CP'},
                {'Range': self.eval_ranges[r], 'Miscoverage': self.avg_miscov_condcp[r + 1], 'Method': 'Centralized CondCP'},
                {'Range': self.eval_ranges[r], 'Miscoverage': self.avg_miscov_naive[r + 1],
                 'Method': 'Centralized GC-FCP'},
                {'Range': self.eval_ranges[r], 'Miscoverage': self.avg_miscov_fedcp[r + 1], 'Method': 'FedCP'},
                {'Range': self.eval_ranges[r], 'Miscoverage': self.avg_miscov_gs[r + 1], 'Method': 'GC-FCP'},  #
            ])

        df = pd.DataFrame(coverage_data)
        df['Method'] = pd.Categorical(
            df['Method'],
            categories=['Centralized CP', 'Centralized CondCP', 'Centralized GC-FCP', 'FedCP', 'GC-FCP'],
            ordered=True
        )
        df['Range'] = pd.Categorical(df['Range'], categories=['Marginal'] + self.eval_ranges, ordered=True)
        return df

    def plot_results(self, save_path: str = None) -> None:
        """Create and display the results visualization."""
        if (self.avg_miscov_vanilla is None or self.avg_miscov_fedcp is None or self.avg_miscov_condcp is None
                or self.avg_miscov_naive is None or self.avg_miscov_gs is None):
            raise ValueError("Must run Monte Carlo simulations first")

        # Generate visualization data
        viz_data = self.generate_visualization_data()

        # Sort data for plotting
        sort_order = np.argsort(viz_data['X_test'])
        X_test_s = viz_data['X_test'][sort_order]
        Y_test_s = viz_data['Y_test'][sort_order]
        Y_test_hat_s = viz_data['Y_test_hat'][sort_order]
        lbs_vanilla_s = viz_data['lbs_vanilla'][sort_order]
        ubs_vanilla_s = viz_data['ubs_vanilla'][sort_order]
        lbs_naive_s = viz_data['lbs_naive'][sort_order]
        ubs_naive_s = viz_data['ubs_naive'][sort_order]
        lbs_fedcp_s = viz_data['lbs_fedcp'][sort_order]
        ubs_fedcp_s = viz_data['ubs_fedcp'][sort_order]
        lbs_condcp_s = viz_data['lbs_condcp'][sort_order]
        ubs_condcp_s = viz_data['ubs_condcp'][sort_order]
        lbs_gs_s = viz_data['lbs_gs'][sort_order]  #
        ubs_gs_s = viz_data['ubs_gs'][sort_order]  #

        # Create coverage DataFrame
        coverage_df = self.create_coverage_dataframe()

        # Set up the plot
        cp = sns.color_palette()
        sns.set(font="DejaVu Sans")
        sns.set_style("whitegrid", {'axes.grid': False})
        fig = plt.figure(figsize=(17.5, 6))

        # Centralized CP plot
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.plot(X_test_s, Y_test_s, '.', alpha=0.5, label='test point')
        ax1.plot(X_test_s, Y_test_hat_s, lw=1, color='k')
        ax1.plot(X_test_s, ubs_vanilla_s, color=cp[0], lw=2)
        ax1.plot(X_test_s, lbs_vanilla_s, color=cp[0], lw=2)
        ax1.fill_between(X_test_s, lbs_vanilla_s, ubs_vanilla_s,
                         color=cp[0], alpha=0.4, label='Centralized CP')
        ax1.plot(X_test_s, ubs_fedcp_s, color=cp[3], lw=2, linestyle=':')
        ax1.plot(X_test_s, lbs_fedcp_s, color=cp[3], lw=2, linestyle=':')
        ax1.fill_between(X_test_s, lbs_fedcp_s, ubs_fedcp_s,
                         color=cp[3], alpha=0.15, label='FedCP')
        ax1.set_ylim(-2, 4)
        ax1.tick_params(axis='both', which='major', labelsize=14)
        ax1.set_xlabel("$X$", fontsize=16, labelpad=10)
        ax1.set_ylabel("$Y$", fontsize=16, labelpad=10)
        # ax1.set_title("Centralized CP", fontsize=18, pad=12)
        ax1.axvspan(1, 2, facecolor='grey', alpha=0.25)
        ax1.axvspan(3, 4, facecolor='grey', alpha=0.25)
        ax1.legend(fontsize=14, loc='upper right')  # legend

        # CondDCP plot
        ax2 = fig.add_subplot(1, 3, 2, sharex=ax1, sharey=ax1)
        ax2.plot(X_test_s, Y_test_s, '.', alpha=0.5, label='test point')
        ax2.plot(X_test_s, Y_test_hat_s, color='k', lw=1)
        # Overlay GC-FCP (, using another color)
        ax2.plot(X_test_s, ubs_condcp_s, color=cp[4], lw=2, linestyle='-.')
        ax2.plot(X_test_s, lbs_condcp_s, color=cp[4], lw=2, linestyle='-.')
        ax2.fill_between(X_test_s, lbs_condcp_s, ubs_condcp_s,
                         color=cp[4], alpha=0.15, label='Centralized CondCP')
        ax2.plot(X_test_s, ubs_naive_s, color=cp[2], lw=2, linestyle='--')
        ax2.plot(X_test_s, lbs_naive_s, color=cp[2], lw=2, linestyle='--')
        ax2.fill_between(X_test_s, lbs_naive_s, ubs_naive_s,
                         color=cp[2], alpha=0.2, label='Centralized GC-FCP')
        ax2.plot(X_test_s, ubs_gs_s, color=cp[1], lw=2)
        ax2.plot(X_test_s, lbs_gs_s, color=cp[1], lw=2)
        ax2.fill_between(X_test_s, lbs_gs_s, ubs_gs_s,
                         color=cp[1], alpha=0.4, label='GC-FCP')

        ax2.tick_params(axis='both', which='major', direction='out', labelsize=14)
        ax2.set_xlabel("$X$", fontsize=16, labelpad=10)
        ax2.set_ylabel("$Y$", fontsize=16, labelpad=10)
        # ax2.set_title("Centralized & GC-FCP", fontsize=18, pad=12)
        ax2.axvspan(1, 2, facecolor='grey', alpha=0.25)
        ax2.axvspan(3, 4, facecolor='grey', alpha=0.25)
        ax2.legend(fontsize=14, loc='upper right')  #  legend

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
        ax3.legend(fontsize=14, loc='upper right')

        plt.tight_layout(pad=3)

        if save_path:
            plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
            plt.savefig(f'{save_path}.pdf', bbox_inches='tight')

        # plt.show()

    def print_results(self) -> None:
        """Print summary of results."""
        if (self.avg_miscov_vanilla is None or self.avg_miscov_fedcp is None or self.avg_miscov_condcp is None
                or self.avg_miscov_naive is None or self.avg_miscov_gs is None):
            raise ValueError("Must run Monte Carlo simulations first")

        print()
        print("=" * 80)
        print("CONFORMAL PREDICTION EXPERIMENT RESULTS (OVERLAPPING GROUPS WITH GC-FCP)")
        print("=" * 80)
        print(f"Number of Monte Carlo simulations: {self.num_mc}")
        print(f"Significance level (α): {self.alpha}")
        print(f"Target coverage: {1 - self.alpha:.1%}")
        print(f"Number of groups: {self.K}")
        print(f"Test samples: {self.n_test}")
        print(f"GC-FCP compression: {self.compression}")

        print()
        print("MISCOVERAGE BY EVALUATION RANGE:")
        print("-" * 80)
        for i, range_name in enumerate(self.eval_ranges):
            vanilla_miscov = self.avg_miscov_vanilla[i + 1]
            fedcp_miscov = self.avg_miscov_fedcp[i + 1]
            condcp_miscov = self.avg_miscov_condcp[i + 1]
            centralized_miscov = self.avg_miscov_naive[i + 1]
            gcfcp_miscov = self.avg_miscov_gs[i + 1]
            print(
                f"{range_name:8} | Vanilla: {vanilla_miscov:.3f} | FedCP: {fedcp_miscov:.3f} | "
                f"CondCP: {condcp_miscov:.3f} | Centralized: {centralized_miscov:.3f} | GC-FCP: {gcfcp_miscov:.3f}"
            )

        print()
        print("MARGINAL MISCOVERAGE:")
        print("-" * 80)
        print(f"Vanilla:     {self.avg_miscov_vanilla[0]:.3f}")
        print(f"FedCP:       {self.avg_miscov_fedcp[0]:.3f}")
        print(f"CondCP:      {self.avg_miscov_condcp[0]:.3f}")
        print(f"Centralized: {self.avg_miscov_naive[0]:.3f}")
        print(f"GC-FCP:      {self.avg_miscov_gs[0]:.3f}")

        print()
        print("STANDARD ERRORS:")
        print("-" * 80)
        vanilla_se = np.std(self.miscov_vanilla_mc, axis=0) / np.sqrt(self.num_mc)
        fedcp_se = np.std(self.miscov_fedcp_mc, axis=0) / np.sqrt(self.num_mc)
        condcp_se = np.std(self.miscov_condcp_mc, axis=0) / np.sqrt(self.num_mc)
        centralized_se = np.std(self.miscov_naive_mc, axis=0) / np.sqrt(self.num_mc)
        gcfcp_se = np.std(self.miscov_gs_mc, axis=0) / np.sqrt(self.num_mc)

        for i, range_name in enumerate(self.eval_ranges):
            print(
                f"{range_name:8} | Vanilla: {vanilla_se[i + 1]:.4f} | FedCP: {fedcp_se[i + 1]:.4f} | "
                f"CondCP: {condcp_se[i + 1]:.4f} | Centralized: {centralized_se[i + 1]:.4f} | GC-FCP: {gcfcp_se[i + 1]:.4f}"
            )

def main():
    """Main execution function."""
    # Initialize experiment
    trials = 100
    K = 4
    n_cal = 2000
    n_test = 200
    n_k = np.array([n_cal * 0.5] + [(0.5 * n_cal) // (K - 1)] * (K - 1), dtype=int)
    pi_k = [1 / K] * K
    alpha = 0.1
    label_shifts = True

    experiment = SyntheticDataDualExperiment(
        K=K,
        label_shifts=label_shifts,
        n_k=n_k,
        pi_k=pi_k,
        alpha=alpha,
        n_test=n_test,
        bounds=(0, 5),
        poly_degree=4,
        num_mc=trials,  # Increased for better statistics
        n_jobs=5,  # Use available cores
        delta=0.1,  # GC-FCP parameter
    )

    # Run Monte Carlo simulations
    experiment.run_monte_carlo()

    # Print results
    experiment.print_results()

    # Create visualization
    save_path = 'figures/synthetic_data_dual_CondDCP_with_GS' if label_shifts else 'figures/synthetic_data_dual_CondDCP_with_GS_covariate'
    # save_path = None
    experiment.plot_results(save_path=save_path)

if __name__ == "__main__":
    main()
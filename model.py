"""
Bi-objective (latency-energy) MEC model, extended from Elkawkagy et al., IEEE Access 2025
("Genetic algorithm-driven joint optimization of task offloading and resource allocation ...").

Changes with respect to the base study
--------------------------------------
1. Separate symbols: f_loc[m] (local CPU frequency, with DVFS) and F[m] (MES CPU share of UD m).
2. Bandwidth-dependent SNR: SNR_m(B) = gamma_m * (B_ref / B), where gamma_m is the SNR measured
   on the reference band B_ref = 1 MHz (equivalent to p*h/(N0*B)).
3. Consistent units: TS (bytes), NC (cycles/byte), R (bit/s) -> bits = 8*TS.
4. Device energy model: E = kappa*f_loc^2*(1-P)*C + p_tx*T_trans.
5. The DVFS variable f_loc creates a genuine latency-energy trade-off.

Encoding (4M genes in [0, 1]; the weights have a small positive floor):
    x[0:M]    P_m      offloaded fraction
    x[M:2M]   wF_m     weights that split F_max (normalized onto the simplex -> constraint C2 always holds)
    x[2M:3M]  wB_m     weights that split B_max (normalized onto the simplex -> constraint C3 always holds)
    x[3M:4M]  u_m      f_loc = f_min + u*(f_max_m - f_min)

Objectives:  f1 = max_m T_m   (seconds, minimax completion time - fairness)
             f2 = sum_m E_m   (joules, device energy)
"""
from dataclasses import dataclass, field
import numpy as np
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair

W_MIN = 1e-6  # weight floor that avoids division by zero


@dataclass
class Scenario:
    M: int = 10
    F_max: float = 10e9          # Hz (cycles/s), total MES CPU capacity
    B_max: float = 20e6          # Hz, total uplink bandwidth (FDMA)
    NC: float = 1000.0           # cycle/byte
    TS_range: tuple = (8e6, 12e6)        # bytes (8-12 MB)
    fmax_range: tuple = (0.8e9, 1.2e9)   # Hz, local CPU ceiling of each UD
    f_min: float = 0.2e9                 # Hz, DVFS floor
    gamma_db_range: tuple = (5.0, 25.0)  # SNR on B_ref (dB)
    B_ref: float = 1e6
    p_tx: float = 0.2            # W
    kappa: float = 1e-27         # effective switched capacitance
    seed: int = 2026
    channel: str = "uniform"     # "uniform": gamma ~ U[5,25] dB as in the base study; "pathloss": see __post_init__
    radius: float = 500.0        # m, cell radius of the "pathloss" channel
    # randomly drawn UD parameters
    TS: np.ndarray = field(init=False)
    C: np.ndarray = field(init=False)
    bits: np.ndarray = field(init=False)
    f_maxm: np.ndarray = field(init=False)
    gamma: np.ndarray = field(init=False)

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        M = self.M
        self.TS = rng.uniform(*self.TS_range, M)
        self.C = self.TS * self.NC
        self.bits = 8.0 * self.TS
        self.f_maxm = rng.uniform(*self.fmax_range, M)
        self.gamma = 10 ** (rng.uniform(*self.gamma_db_range, M) / 10)
        if self.channel == "pathloss":
            # UDs uniform in a disc (35 m <= d <= radius), path loss 128.1 + 37.6 log10(d/km) dB,
            # Rayleigh block fading |h|^2 ~ Exp(1) drawn once, noise -174 dBm/Hz with a 9 dB noise figure
            d = np.sqrt(rng.uniform((35.0 / self.radius) ** 2, 1.0, M)) * self.radius
            pl_db = 128.1 + 37.6 * np.log10(d / 1000.0)
            fade = rng.exponential(1.0, M)
            p_dbm = 10 * np.log10(self.p_tx * 1000.0)
            noise_dbm = -174.0 + 10 * np.log10(self.B_ref) + 9.0
            self.gamma = 10 ** ((p_dbm - pl_db - noise_dbm) / 10) * fade
            self.dist = d


def decode(X, sc: Scenario):
    """X: (n, 4M) -> physical variables P, F, B, f, each of shape (n, M)."""
    X = np.atleast_2d(X)
    M = sc.M
    P = np.clip(X[:, :M], 0.0, 1.0)
    wF = np.maximum(X[:, M:2 * M], W_MIN)
    wB = np.maximum(X[:, 2 * M:3 * M], W_MIN)
    u = np.clip(X[:, 3 * M:], 0.0, 1.0)
    F = sc.F_max * wF / wF.sum(1, keepdims=True)
    B = sc.B_max * wB / wB.sum(1, keepdims=True)
    f = sc.f_min + u * (sc.f_maxm - sc.f_min)
    return P, F, B, f


def components(X, sc: Scenario):
    P, F, B, f = decode(X, sc)
    R = B * np.log2(1.0 + sc.gamma * sc.B_ref / B)
    T_loc = (1 - P) * sc.C / f
    T_tx = P * sc.bits / R
    T_mec = P * sc.C / F
    T = np.maximum(T_loc, T_tx + T_mec)
    E = sc.kappa * f ** 2 * (1 - P) * sc.C + sc.p_tx * T_tx
    return dict(P=P, F=F, B=B, f=f, R=R, T_loc=T_loc, T_tx=T_tx,
                T_mec=T_mec, T=T, E=E)


def objectives(X, sc: Scenario):
    c = components(X, sc)
    return np.column_stack([c["T"].max(1), c["E"].sum(1)])


class MECProblem(Problem):
    def __init__(self, sc: Scenario):
        self.sc = sc
        n = 4 * sc.M
        xl = np.zeros(n)
        xl[sc.M:3 * sc.M] = W_MIN
        super().__init__(n_var=n, n_obj=2, xl=xl, xu=np.ones(n))

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = objectives(X, self.sc)


class SlackRepair(Repair):
    """Pareto-safe DVFS slack repair (Proposition 3 of the paper).

    For a solution x, let T* = max_m T_m. Every UD whose local part finishes before T* lowers
    f_loc to the smallest sufficient value: f = max(f_min, (1-P)C / T*).
    -> f1 is unchanged and f2 decreases or stays the same, so the new solution weakly dominates the old one.
    """

    def _do(self, problem, X, **kwargs):
        sc = problem.sc
        M = sc.M
        X = X.copy()
        c = components(X, sc)
        Tstar = c["T"].max(1, keepdims=True)
        f_need = (1 - c["P"]) * sc.C / Tstar
        f_new = np.clip(np.maximum(sc.f_min, f_need), sc.f_min, c["f"])
        X[:, 3 * M:] = (f_new - sc.f_min) / (sc.f_maxm - sc.f_min)
        return X


# ---------------------------------------------------------------------------
# Exact latency-optimal solver: bisection on T + convex subproblem (Algorithm 1, Proposition 1)
# ---------------------------------------------------------------------------
def _bisect(fun, lo, hi, it=100):
    """Root of an increasing function fun on [lo, hi] (vectorized)."""
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        pos = fun(mid) > 0
        hi = np.where(pos, mid, hi); lo = np.where(pos, lo, mid)
    return 0.5 * (lo + hi)


def _min_F_needed(T, sc):
    """For a latency level T (f_loc = f_max), the minimum total MES capacity F that achieves T.
    P_m = max(0, 1 - T/a_m): smallest offloaded fraction with local part <= T (Lemma 2).
    Need tx_m(B_m) + C_m/F_m <= s_m = T/P_m  ->  F_m = h_m(B_m) = C_m/(s_m - tx_m(B_m)).
    min sum h_m(B_m) s.t. sum B_m = B_max is a separable convex program (Lemma 3); its KKT
    conditions are solved exactly by nested bisections on the Lagrange multiplier lambda."""
    a = sc.C / sc.f_maxm
    P = np.maximum(0.0, 1.0 - T / a)
    act = P > 1e-12
    M = sc.M
    F = np.zeros(M); B = np.zeros(M)
    if not act.any():
        return 0.0, P, F, B
    s, bits, C, k = T / P[act], sc.bits[act], sc.C[act], sc.gamma[act] * sc.B_ref
    rate = lambda Bv: Bv * np.log2(1 + k / Bv)
    tx = lambda Bv: bits / rate(Bv)
    def dh(Bv):  # derivative of h_m with respect to B (negative, increasing towards 0)
        r = rate(Bv)
        dr = np.log2(1 + k / Bv) - (k / Bv) / ((1 + k / Bv) * np.log(2))
        return C * (-bits * dr / r ** 2) / (s - tx(Bv)) ** 2
    Bmax = sc.B_max
    # smallest bandwidth that meets the deadline: tx(B_lo) = s
    B_lo = _bisect(lambda Bv: s - tx(Bv), np.full(k.shape, 1e-3), np.full(k.shape, 1e3 * Bmax))
    if B_lo.sum() >= Bmax:
        return np.inf, P, F, B
    B_lo = B_lo * (1 + 1e-9)
    def B_of(lam):
        lam = np.full(k.shape, lam)
        Bv = _bisect(lambda Bv: dh(Bv) + lam, B_lo, np.full(k.shape, Bmax))
        return Bv
    # large lambda -> small B; find lambda with sum B = Bmax (bisection on log lambda)
    llo, lhi = -40.0, 40.0
    for _ in range(100):
        lm = 0.5 * (llo + lhi)
        if B_of(10 ** lm).sum() > Bmax:
            llo = lm
        else:
            lhi = lm
    Bv = B_of(10 ** lhi)
    Bv = Bv * Bmax / Bv.sum() if Bv.sum() > Bmax else Bv
    F[act] = C / (s - tx(Bv)); B[act] = Bv
    return F.sum(), P, F, B


_CACHE = {}


def latency_optimal(sc: Scenario, tol=1e-5):
    key = (sc.M, sc.F_max, sc.B_max, sc.seed, tol)
    if key not in _CACHE:
        _CACHE[key] = _latency_optimal(sc, tol)
    return _CACHE[key]


def _latency_optimal(sc: Scenario, tol):
    """Return (T*, x*): the globally optimal minimax latency (within tol seconds) and an encoded solution."""
    lo, hi = 0.0, float((sc.C / sc.f_maxm).max())  # hi: all-local execution is always feasible
    best = None
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        tot, P, F, B = _min_F_needed(mid, sc)
        if tot <= sc.F_max:
            hi, best = mid, (P, F, B)
        else:
            lo = mid
    P, F, B = best if best else (np.zeros(sc.M), np.full(sc.M, sc.F_max / sc.M),
                                 np.full(sc.M, sc.B_max / sc.M))
    # distribute any spare F evenly, then encode as genes
    F = np.where(F > 0, F, 0) + (sc.F_max - F.sum()) / sc.M
    x = np.concatenate([P, np.maximum(F / F.max(), W_MIN),
                        np.maximum(B / B.max(), W_MIN), np.ones(sc.M)])
    return hi, x


def seed_population(sc: Scenario, n, rng):
    """Initial population seeded with special solutions:
    - the latency-optimal solution (extreme point of f1)
    - all-local execution at f_min (simple energy-side seed)
    - random solutions for the rest."""
    M = sc.M
    X = rng.random((n, 4 * M))
    X[:, M:3 * M] = np.maximum(X[:, M:3 * M], W_MIN)
    _, x_lat = latency_optimal(sc)
    x_en = np.concatenate([np.zeros(M), np.ones(2 * M), np.zeros(M)])
    X[0], X[1] = x_lat, x_en
    return X

"""
Extensions of model.py used by the manuscript (introduced in v3):
  - energy_optimal(sc): exact minimum of f2 (Proposition 2 of the paper)
  - seed_population_exact(sc, n, rng): seeding with BOTH exact anchors
    (exact latency optimum x_lat and exact energy optimum x_en*).
The original model.py is left unchanged.
"""
import itertools, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Scenario, objectives, latency_optimal, W_MIN  # noqa: E402

LN2 = np.log(2.0)


def _rate(B, k):
    return B * np.log2(1 + k / B)


FAST = True   # bisection on a logarithmic scale with tolerance-based iteration counts (same algorithm)


def _log_bisect(pos_fun, lo, hi, it):
    """Vectorized bisection on log(x): pos_fun(x) is True to the left of the root."""
    a, b = np.log(lo), np.log(hi)
    for _ in range(it):
        mid = 0.5 * (a + b)
        pos = pos_fun(np.exp(mid))
        a = np.where(pos, mid, a); b = np.where(pos, b, mid)
    return np.exp(0.5 * (a + b))


def _tx_energy_split(sc, S):
    """min sum_{m in S} p_tx D_m / R_m(B_m)  s.t. sum B_m = B_max  (convex, KKT).
    Vectorized nested bisection: inner on each B_m, outer on log10(lambda)."""
    k = sc.gamma[S] * sc.B_ref
    D = sc.bits[S]
    def neg_grad(B):  # -d/dB [p D / R(B)] > 0 and decreasing in B (convexity)
        dr = np.log2(1 + k / B) - (k / B) / ((1 + k / B) * LN2)
        return sc.p_tx * D * dr / _rate(B, k) ** 2
    if FAST:
        # relative precision of B: ln(B_max/1e-6)/2^44 < 2e-12; of lambda: 40 decades / 2^46 < 1e-12
        B_of = lambda lam: _log_bisect(lambda B: neg_grad(B) > lam, np.full(k.shape, 1e-6), np.full(k.shape, sc.B_max), 44)
        n_out = 46
    else:
        def B_of(lam):
            lo, hi = np.full(k.shape, 1e-6), np.full(k.shape, sc.B_max)
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                pos = neg_grad(mid) > lam
                lo = np.where(pos, mid, lo); hi = np.where(pos, hi, mid)
            return 0.5 * (lo + hi)
        n_out = 80
    lo, hi = -30.0, 10.0
    for _ in range(n_out):
        mid = 0.5 * (lo + hi)
        if B_of(10 ** mid).sum() > sc.B_max:
            lo = mid
        else:
            hi = mid
    B = B_of(10 ** hi)
    B *= sc.B_max / B.sum()
    return float(np.sum(sc.p_tx * D / _rate(B, k))), B


# ---------------------------------------------------------------------------- fast latency anchor
def _min_F_needed_fast(T, sc):
    """Same computation as model._min_F_needed (Lemmas 2 and 3, Proposition 1), with log-scale bisections
    whose iteration counts follow from the required precision instead of a fixed 100."""
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
    def dh(Bv):
        r = rate(Bv)
        dr = np.log2(1 + k / Bv) - (k / Bv) / ((1 + k / Bv) * LN2)
        return C * (-bits * dr / r ** 2) / (s - tx(Bv)) ** 2
    Bmax = sc.B_max
    B_lo = _log_bisect(lambda Bv: s - tx(Bv) <= 0, np.full(k.shape, 1e-3), np.full(k.shape, 1e3 * Bmax), 50)
    if B_lo.sum() >= Bmax:
        return np.inf, P, F, B
    B_lo = B_lo * (1 + 1e-9)
    B_of = lambda lam: _log_bisect(lambda Bv: dh(Bv) + lam < 0, B_lo, np.full(k.shape, Bmax), 44)
    llo, lhi = -40.0, 40.0
    for _ in range(48):
        lm = 0.5 * (llo + lhi)
        if B_of(10 ** lm).sum() > Bmax:
            llo = lm
        else:
            lhi = lm
    Bv = B_of(10 ** lhi)
    Bv = Bv * Bmax / Bv.sum() if Bv.sum() > Bmax else Bv
    F[act] = C / (s - tx(Bv)); B[act] = Bv
    return F.sum(), P, F, B


def latency_optimal_fast(sc: Scenario, tol=1e-5):
    """Algorithm 1 with the fast inner solver; shares the cache of model.latency_optimal."""
    import model as _m
    key = (sc.M, sc.F_max, sc.B_max, sc.seed, tol)
    if key in _m._CACHE:
        return _m._CACHE[key]
    lo, hi = 0.0, float((sc.C / sc.f_maxm).max())
    best = None
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        tot, P, F, B = _min_F_needed_fast(mid, sc)
        if tot <= sc.F_max:
            hi, best = mid, (P, F, B)
        else:
            lo = mid
    P, F, B = best if best else (np.zeros(sc.M), np.full(sc.M, sc.F_max / sc.M), np.full(sc.M, sc.B_max / sc.M))
    F = np.where(F > 0, F, 0) + (sc.F_max - F.sum()) / sc.M
    x = np.concatenate([P, np.maximum(F / F.max(), W_MIN), np.maximum(B / B.max(), W_MIN), np.ones(sc.M)])
    _m._CACHE[key] = (hi, x)
    return hi, x


# every module that calls model.latency_optimal after importing model_ext uses the fast solver
# (identical anchors, see check_fast_anchors.py)
import model as _model_module  # noqa: E402
if FAST:
    _model_module.latency_optimal = latency_optimal_fast
    latency_optimal = latency_optimal_fast


_E_CACHE = {}


def energy_optimal(sc: Scenario):
    key = (sc.M, sc.F_max, sc.B_max, sc.seed)
    if key not in _E_CACHE:
        _E_CACHE[key] = _energy_optimal(sc)
    return _E_CACHE[key]


def break_even_bandwidth(sc, m):
    """b_m: the bandwidth at which transmitting the whole task of UD m costs as much energy as
    executing it locally at f_min (infinite if even the whole band is not enough)."""
    e_loc = sc.kappa * sc.f_min ** 2 * sc.C[m]
    k = sc.gamma[m] * sc.B_ref
    if sc.p_tx * sc.bits[m] / _rate(sc.B_max, k) >= e_loc:
        return np.inf
    lo, hi = 1e-3, sc.B_max
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if sc.p_tx * sc.bits[m] / _rate(mid, k) > e_loc:
            lo = mid
        else:
            hi = mid
    return hi


def _energy_optimal(sc: Scenario):
    """Exact E* and an encoded solution x_en* (Proposition 2).
    f_m = f_min, P_m in {0,1}. An offloaded UD must receive more than its break-even bandwidth b_m
    (otherwise computing locally is not worse and frees bandwidth), so an optimal offloading set O
    satisfies sum_{m in O} b_m < B_max. Only these sets are enumerated, by a depth-first search over
    the candidates sorted by b_m."""
    M = sc.M
    e_loc = sc.kappa * sc.f_min ** 2 * sc.C
    b = np.array([break_even_bandwidth(sc, m) for m in range(M)])
    cand = [int(m) for m in np.argsort(b) if np.isfinite(b[m])]
    best_E, best_S, best_B = float(e_loc.sum()), (), None
    n_sets = 0

    def dfs(i, S, used):
        nonlocal best_E, best_S, best_B, n_sets
        for j in range(i, len(cand)):
            m = cand[j]
            if used + b[m] >= sc.B_max:
                break                                   # candidates are sorted by b_m
            T = S + [m]
            n_sets += 1
            e_tx, B = _tx_energy_split(sc, T)
            E = float(e_loc.sum() - e_loc[T].sum() + e_tx)
            if E < best_E:
                best_E, best_S, best_B = E, tuple(T), B
            dfs(j + 1, T, used + b[m])

    dfs(0, [], 0.0)
    x = np.concatenate([np.zeros(M), np.full(M, W_MIN), np.full(M, W_MIN), np.zeros(M)])
    if best_S:
        S = list(best_S)
        x[S] = 1.0                                  # P_m = 1
        x[M + np.array(S)] = 1.0                    # all MES CPU to offloaded UDs
        x[2 * M + np.array(S)] = best_B / best_B.max()
    else:
        x[M:3 * M] = 1.0
    return best_E, x, dict(candidates=cand, offload_set=list(best_S), subsets=n_sets,
                           max_set=int(np.floor(sc.B_max / b[cand[0]])) if cand else 0)


def seed_population_exact(sc: Scenario, n, rng):
    """Initial population with both exact anchors + (n-2) random vectors."""
    M = sc.M
    X = rng.random((n, 4 * M))
    X[:, M:3 * M] = np.maximum(X[:, M:3 * M], W_MIN)
    _, x_lat = latency_optimal(sc)
    _, x_en, _ = energy_optimal(sc)
    X[0], X[1] = x_lat, x_en
    return X


if __name__ == "__main__":
    for M in (10, 20, 30):
        sc = Scenario(M=M)
        E, x, info = energy_optimal(sc)
        print(f"M={M}: E*={E:.4f} J, all-local={np.sum(sc.kappa*sc.f_min**2*sc.C):.4f} J, "
              f"check f(x)={objectives(x[None], sc)[0]}, candidates={info['candidates']}, set={info['offload_set']}")

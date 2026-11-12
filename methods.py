"""
All methods compared in the paper, under a common budget of 30 000 evaluations.

Proposed: ACE-SQP (Anchored Continuation Epsilon-constraint SQP): both exact anchors plus a ladder of
epsilon-constraint problems solved by SLSQP with closed-form gradients, warm-started level by level
from the latency anchor, with an activation restart that escapes the P_m = 0 trap; every solution is
slack-repaired.

The exact anchors are computed once per scenario, reported separately and shared through a file.
"""
import os, pickle, time
import numpy as np
from scipy.optimize import minimize as sp_minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.callback import Callback
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.ref_dirs import get_reference_directions

import model as mdl
import model_ext as ext
from model import Scenario, MECProblem, SlackRepair, objectives, latency_optimal, W_MIN
from model_ext import energy_optimal, seed_population_exact
from sqp_core import (Counter, BudgetExceeded, bounds_z, model, energy_grad, time_jacobians, budget_jac, budget_values,
                      to_genes, from_genes, solve_eps, eps_levels, FTOL, RESTARTS)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
CACHE = os.path.join(RESULTS, "anchors_cache.pkl")
POP, NGEN = 100, 300
TOTAL = POP * NGEN
K_FRONT = POP - 2             # with both anchors the front has at most POP points, like the MOEAs
PROP = "ACE-SQP"


# ---------------------------------------------------------------------------- anchors
def anchor_job(kw):
    sc = Scenario(**kw)
    t0 = time.time(); T, x_lat = latency_optimal(sc); t_lat = time.time() - t0
    t0 = time.time(); E, x_en, info = energy_optimal(sc); t_en = time.time() - t0
    return kw, dict(T=T, x_lat=x_lat, E=E, x_en=x_en, info=info, t_lat=t_lat, t_en=t_en)


def key_of(kw):
    return tuple(sorted(kw.items()))


def load_cache():
    return pickle.load(open(CACHE, "rb")) if os.path.exists(CACHE) else {}


def ensure_anchors(kws, pool):
    cache = load_cache()
    need = [kw for kw in kws if key_of(kw) not in cache]
    for k, v in pool.imap_unordered(anchor_job, need):
        cache[key_of(k)] = v
        pickle.dump(cache, open(CACHE, "wb"))
    return cache


def init_worker():
    """Fill the in-memory caches of model.py and model_ext.py from the shared file."""
    for k, a in load_cache().items():
        sc = Scenario(**dict(k))
        mdl._CACHE[(sc.M, sc.F_max, sc.B_max, sc.seed, 1e-5)] = (a["T"], a["x_lat"])
        ext._E_CACHE[(sc.M, sc.F_max, sc.B_max, sc.seed)] = (a["E"], a["x_en"], a["info"])


# ---------------------------------------------------------------------------- helpers
def nondom(F):
    F = np.asarray(F)
    idx = NonDominatedSorting().do(F, only_non_dominated_front=True)
    F = F[idx]
    return F[np.argsort(F[:, 0])]


def t_nadir(sc):
    return float(np.max(sc.C / sc.f_min))


class Trace(Callback):
    def __init__(self):
        super().__init__(); self.hist = []

    def notify(self, algorithm):
        if algorithm.n_gen % 10 == 0 or algorithm.n_gen == 1:
            self.hist.append((algorithm.evaluator.n_eval, algorithm.opt.get("F").copy()))


def ops_kwargs(ops):
    """Crossover and mutation settings; None keeps the pymoo defaults."""
    kw = {}
    if ops:
        kw["crossover"] = SBX(eta=ops["sbx_eta"], prob=0.9)
        kw["mutation"] = PM(eta=ops["pm_eta"])
    return kw


# ---------------------------------------------------------------------------- the proposed method
def activated_start(z, sc, tol=1e-3):
    """Activation restart. A UD with P_m = 0 and B_m at its floor sits in a trap: the transmission energy
    per offloaded bit, p D / R(B_m), is so large there that the gradient keeps P_m at zero. Give every
    inactive UD the median offloaded fraction, bandwidth and MES share of the active UDs and rescale the
    budgets. Returns None when no UD is inactive or none is active."""
    M_ = sc.M
    P, f, F, B = z[:M_].copy(), z[M_:2 * M_].copy(), z[2 * M_:3 * M_].copy(), z[3 * M_:4 * M_].copy()
    off = P < tol
    if not off.any() or off.all():
        return None
    P[off], F[off], B[off] = np.median(P[~off]), np.median(F[~off]), np.median(B[~off])
    lb, ub = bounds_z(sc)
    F *= (sc.F_max / 1e9) / F.sum()
    B *= (sc.B_max / 1e6) / B.sum()
    return np.clip(np.concatenate([P, f, F, B]), lb, ub)


def random_z(sc, rng):
    """A random decision vector (genes uniform in [0, 1], shares at least W_MIN), as the random starts of WS-SQP."""
    x = rng.random(4 * sc.M)
    x[sc.M:3 * sc.M] = np.maximum(x[sc.M:3 * sc.M], W_MIN)
    return from_genes(x, sc)


def restart_start(z, sc, mode, rng):
    """Start of the restart of ACE-SQP (v13, reviewer item: is the activation restart better than a plain restart?).
    mode "activation": the activated start of z; mode "random": a random point, used under exactly the same condition
    (some UD is idle in z) and back-off, so that only the content of the restart differs."""
    za = activated_start(z, sc)
    if za is None or mode == "activation":
        return za
    return random_z(sc, rng)


def sqp_ladder(sc, budget=TOTAL, K=K_FRONT, rho=1.0, shape=2.0, use_grad=True, repair=True,
               continuation=True, cold_seed=None, safeguard=True, backward=False, activation=True, borrow=1.0,
               restart="activation", restart_seed=0):
    """Solve K epsilon-constraint problems by SLSQP.
    rho: eps_max = T* + rho (T_nad - T*); shape: exponent of the level spacing.
    Forward sweep from the latency anchor; backward=True adds a sweep from the energy anchor.
    safeguard: a solution of level i stays feasible at level i+1, so the energy of the kept solution
    never increases with epsilon, and a failed solve cannot derail the sweep.
    continuation=False restarts every level from the latency anchor; cold_seed starts from a random point.
    borrow: allowance of a solve = min(remaining budget, borrow x remaining budget / remaining solves); borrow = 1 is
    the even split of the first version, borrow > 1 lets a hard level use budget that later, warm-started levels
    usually do not need (selected on the development scenarios, see tune_borrow.py)."""
    T_star, x_lat = latency_optimal(sc)
    E_ref = float(objectives(x_lat[None], sc)[0, 1])
    levels = eps_levels(T_star, T_star + rho * (t_nadir(sc) - T_star), K, shape)
    z_lat = from_genes(x_lat, sc)
    if cold_seed is not None:
        x0 = np.random.default_rng(cold_seed).random(4 * sc.M)
        x0[sc.M:3 * sc.M] = np.maximum(x0[sc.M:3 * sc.M], W_MIN)
        z_lat = from_genes(x0, sc)
    cnt, stats = Counter(limit=budget), []    # hard cap on the evaluations
    best = [None] * K      # (energy, z) of the best feasible solution of each level
    last = [None] * K      # last iterate of each level, used when safeguard is off
    n_rejected, n_act = 0, 0

    def feasible_energy(i, z):
        m = model(z, sc)
        ok = max(m["Tl"].max(), (m["Tt"] + m["Tm"]).max()) <= levels[i] * (1 + 1e-6)
        return ok, float(m["E"])

    def consider(i, z):
        ok, E = feasible_energy(i, z)
        if ok and (best[i] is None or E < best[i][0]):
            best[i] = (E, z.copy())
            return True
        return False

    def solve(i, z0, passes_left):
        left = budget - cnt.n
        if left <= 0:
            return z0
        n0 = cnt.n
        try:
            allow = int(min(left, borrow * left / max(1, passes_left)))
            z, res, nit = solve_eps(z0, levels[i] * (1 + 1e-9), sc, cnt, allow, E_ref, use_grad)
            status = int(res.status)
        except Exception:
            z, status, nit = z0, -1, 0
        m = model(z, sc)
        stats.append(dict(eps=float(levels[i]), status=status, nit=nit, evals=cnt.n - n0,
                          viol=float(max(m["Tl"].max(), (m["Tt"] + m["Tm"]).max()) - levels[i])))
        return z

    total_passes = K * (2 if backward else 1)
    z = z_lat
    rng_r = np.random.default_rng(restart_seed)
    next_act, gap = 1, 1   # activation back-off: after an unsuccessful try the next one is 2, 4, ... levels later
    for i in range(K):
        start = z if continuation else z_lat
        z = solve(i, start, total_passes - i)
        last[i] = z
        consider(i, z)
        z_act = restart_start(z, sc, restart, rng_r) if activation and i >= next_act else None
        if z_act is not None:
            n_act += 1
            if consider(i, solve(i, z_act, total_passes - i)):
                gap = 1
            else:
                gap *= 2
            next_act = i + gap
        if safeguard:
            if i > 0 and best[i - 1] is not None and feasible_energy(i, best[i - 1][1])[1] < (best[i][0] if best[i] else np.inf):
                consider(i, best[i - 1][1])
                n_rejected += 1
            if best[i] is not None:
                z = best[i][1]
    if backward:
        z = from_genes(energy_optimal(sc)[1], sc)
        for i in reversed(range(K)):
            z = solve(i, z, i + 1)
            consider(i, z)
            z_act = activated_start(z, sc) if activation and i < K - 1 else None
            if z_act is not None:
                n_act += 1
                consider(i, solve(i, z_act, i + 1))
            if best[i] is not None:
                z = best[i][1]
        for i in range(1, K):        # energy is non-increasing in epsilon
            if best[i - 1] is not None and (best[i] is None or best[i - 1][0] < best[i][0]):
                consider(i, best[i - 1][1])
    rep, prob = SlackRepair(), MECProblem(sc)
    Z = [b[1] for b in best if b is not None] if (safeguard or backward) else [l for l in last if l is not None]
    X = np.array([to_genes(q, sc) for q in Z])
    if repair and len(X):
        X = rep._do(prob, X)
    return X, cnt.n, dict(levels=stats, n_rejected=n_rejected, n_kept=len(Z), n_activated=n_act)


def run_ace(sc, seed=0, K=K_FRONT, rho=1.0, shape=2.0, use_energy_anchor=True, **kw):
    _, x_lat = latency_optimal(sc)
    X, used, stats = sqp_ladder(sc, K=K, rho=rho, shape=shape, **kw)
    parts = [x_lat[None], X]
    if use_energy_anchor:
        parts.append(energy_optimal(sc)[1][None])
    X = np.vstack(parts)
    return objectives(X, sc), X, used, stats


# ---------------------------------------------------------------------------- hybrid (ACE-SQP + evolutionary refinement)
def hv_truncate(F, X, n, lo, hi):
    """Reduce a non-dominated set (sorted by J1) to at most n points by repeatedly removing the interior point with
    the smallest exclusive hypervolume contribution in the space normalized by lo and hi (exact in two dimensions);
    both extreme points are always kept."""
    G = (F - lo) / (hi - lo)
    keep = list(range(len(F)))
    while len(keep) > n:
        g = G[keep]
        c = (g[2:, 0] - g[1:-1, 0]) * (g[:-2, 1] - g[1:-1, 1])
        del keep[1 + int(np.argmin(c))]
    return F[keep], (X[keep] if X is not None else None)


class Archive(Callback):
    """External archive: every solution evaluated by the evolutionary phase (including the initial population) is
    offered to an unbounded non-dominated archive, so that no non-dominated point is ever lost. hist records the
    archive, reduced to POP points, every ten generations."""
    def __init__(self, lo, hi, record=False):
        super().__init__()
        self.F, self.X, self.lo, self.hi, self.record, self.hist = None, None, lo, hi, record, []

    def add(self, F, X):
        F = F if self.F is None else np.vstack([self.F, F])
        X = X if self.X is None else np.vstack([self.X, X])
        idx = NonDominatedSorting().do(F, only_non_dominated_front=True)
        F, X = F[idx], X[idx]
        F, u = np.unique(F, axis=0, return_index=True)       # sorted by J1; exact duplicates removed
        self.F, self.X = F, X[u]

    def front(self, n=POP):
        return hv_truncate(self.F, self.X, n, self.lo, self.hi)

    def notify(self, algorithm):
        off = algorithm.off
        if off is not None and len(off):
            self.add(off.get("F"), off.get("X"))
        if self.record and (algorithm.n_gen % 10 == 0 or algorithm.n_gen == 1):
            self.hist.append((algorithm.evaluator.n_eval, self.front()[0]))


def run_hybrid(sc, seed, K=12, rho=0.45, shape=2.0, ops=None, record=False, borrow=1.0):
    """ACE-SQP followed by evolutionary refinement: the anchors and the SQP ladder form the initial population of
    SR-NSGA-II, which spends the remaining budget; an external archive keeps every non-dominated solution evaluated,
    and the returned front is the archive reduced to POP points by hypervolume contribution (the same maximum front
    size as every other method). Returns (F, X, used, n_gen, stats, hist)."""
    rng = np.random.default_rng(seed)
    Xs, used, stats = sqp_ladder(sc, K=K, rho=rho, shape=shape, borrow=borrow)
    X0 = seed_population_exact(sc, POP, rng)
    X0[2:2 + len(Xs)] = Xs
    n_gen = max(1, (TOTAL - used) // POP)
    T_star, x_lat = latency_optimal(sc)
    E_star = energy_optimal(sc)[0]
    lo = np.array([T_star, E_star])
    hi = np.array([t_nadir(sc), float(objectives(x_lat[None], sc)[0, 1])])
    arch = Archive(lo, hi, record=record)
    minimize(MECProblem(sc), NSGA2(pop_size=POP, sampling=X0, repair=SlackRepair(), **ops_kwargs(ops)),
             ("n_gen", n_gen), seed=seed, verbose=False, callback=arch)
    F, X = arch.front()
    return F, X, used, n_gen, stats, arch.hist


# ---------------------------------------------------------------------------- weighted-sum baselines
def run_ws_sqp(sc, seed, n_w=K_FRONT):
    """Normalized weighted sums in epigraph form with the same gradients and scaling as ACE-SQP;
    warm start from the previous weight plus one random start per weight."""
    rng = np.random.default_rng(seed)
    M = sc.M
    T_star, x_lat = latency_optimal(sc)
    E_star, _, _ = energy_optimal(sc)
    T_nad, E_nad = t_nadir(sc), float(objectives(x_lat[None], sc)[0, 1])
    lb0, ub0 = bounds_z(sc)
    s = np.append(ub0, T_nad)
    lo, hi = np.append(lb0, 0.0) / s, np.append(ub0, T_nad) / s
    A = budget_jac(sc)
    cnt = Counter(limit=TOTAL)                 # hard cap on the evaluations

    def cons(y):
        cnt.tick(y)
        v = y * s
        m = model(v[:-1], sc)
        return np.concatenate([(v[-1] - m["Tl"]) / T_nad, (v[-1] - m["Tt"] - m["Tm"]) / T_nad, budget_values(v[:-1], sc)])

    def cons_jac(y):
        cnt.tick(y)
        v = y * s
        Jl, Jo = time_jacobians(v[:-1], sc)
        one = np.ones((M, 1))
        J = np.vstack([np.hstack([-Jl / T_nad, one]), np.hstack([-Jo / T_nad, one]), np.hstack([A, np.zeros((2, 1))])])
        J[:, :4 * M] *= s[:4 * M]  # chain rule; the last column is already d/d(t/T_nad)
        return J

    def start(x):
        z = from_genes(x, sc)
        m = model(z, sc)
        t = min(T_nad, float(max(m["Tl"].max(), (m["Tt"] + m["Tm"]).max())) * 1.0001)
        return np.clip(np.append(z, t) / s, lo, hi)

    def rand_start():
        x = rng.random(4 * M); x[M:3 * M] = np.maximum(x[M:3 * M], W_MIN)
        return start(x)

    prev, sols = start(x_lat), []
    per_solve = TOTAL // (2 * n_w)
    for w in np.linspace(1, 0, n_w):
        def obj(y):
            cnt.tick(y)
            v = y * s
            return w * (v[-1] - T_star) / (T_nad - T_star) + (1 - w) * (model(v[:-1], sc)["E"] - E_star) / (E_nad - E_star)

        def grad(y):
            cnt.tick(y)
            v = y * s
            return np.append((1 - w) * energy_grad(v[:-1], sc) * s[:-1] / (E_nad - E_star), w * T_nad / (T_nad - T_star))

        best = None
        for y0 in (prev, rand_start()):
            if cnt.n >= TOTAL:
                break
            try:
                res = sp_minimize(obj, y0, jac=grad, method="SLSQP", bounds=list(zip(lo, hi)),
                                  constraints=[{"type": "ineq", "fun": cons, "jac": cons_jac}],
                                  options=dict(maxiter=max(1, per_solve // 4), ftol=FTOL))
                y = np.clip(res.x, lo, hi)
            except BudgetExceeded:                  # hard cap: keep the last evaluated point
                y = np.clip(cnt.last, lo, hi) if cnt.last is not None else y0
            except Exception:
                y = y0
            F_ = objectives(to_genes((y * s)[:-1], sc)[None], sc)[0]
            val = w * (F_[0] - T_star) / (T_nad - T_star) + (1 - w) * (F_[1] - E_star) / (E_nad - E_star)
            if best is None or val < best[0]:
                best = (val, y, F_)
        if best is not None:
            sols.append(best[2]); prev = best[1]
    return np.array(sols), cnt.n


# ---------------------------------------------------------------------------- deterministic scalarization baselines (v12, reviewer item M5)
def _slsqp_counted(obj, grad, cons, y0, lo, hi, cnt, allow, restarts=RESTARTS, per_iter=4):
    """SLSQP with the evaluation accounting of solve_eps: at most `allow` counted calls, restarts from the last
    iterate when it stops without convergence; returns (y, status)."""
    y, n0, status = y0, cnt.n, 9
    for _ in range(restarts + 1):
        left = allow - (cnt.n - n0)
        if left < per_iter:
            break
        try:
            res = sp_minimize(obj, y, jac=grad, method="SLSQP", bounds=list(zip(lo, hi)), constraints=cons,
                              options=dict(maxiter=max(1, left // per_iter), ftol=FTOL))
            y, status = np.clip(res.x, lo, hi), int(res.status)
        except BudgetExceeded:
            if cnt.last is not None:
                y = np.clip(cnt.last, lo, hi)
            return y, 9
        except Exception:
            return y, -1
        if status == 0:
            break
    return y, status


def _front_from_z(Z, sc, x_lat, x_en):
    """Slack-repaired decisions of the solutions Z (physical vectors) plus both anchors; returns objectives and genes."""
    X = np.array([to_genes(z, sc) for z in Z]) if len(Z) else np.zeros((0, 4 * sc.M))
    if len(X):
        X = SlackRepair()._do(MECProblem(sc), X)
    X = np.vstack([x_lat[None], X, x_en[None]])
    return objectives(X, sc), X


def run_nbi_sqp(sc, seed=0, n_b=K_FRONT, borrow=2.0, restart=None):
    """restart (v13): None (v12 NBI-SQP); "activation": the activation restart of ACE-SQP, with its trigger (an idle
    UD) and back-off; "random": one additional random start for every subproblem, as in WS-SQP (gives NBI-SQP about
    twice the evaluations and run time). The better of the two solutions is kept (feasible first, then the NBI
    objective)."""
    if restart is None:
        return _run_nbi(sc, seed, n_b, borrow, None)
    return _run_nbi(sc, seed, n_b, borrow, restart)


def _run_nbi(sc, seed, n_b, borrow, restart):
    """Normal-boundary intersection [Das and Dennis 1998] with the same closed-form gradients, scaling and solver as
    ACE-SQP. In the objective space normalized by the exact anchors, the anchors are (0, 1) and (1, 0), so the NBI
    subproblem for beta is: minimize T^ + E^ subject to T^ - E^ = 1 - 2 beta, with the minimax latency in epigraph
    form (tau >= T_m^loc, tau >= T_m^tx + T_m^mec). beta = 1 - k/(n_b + 1), k = 1..n_b, from the latency end; each
    subproblem is warm-started from the previous one. Budget allowance as in ACE-SQP."""
    M_ = sc.M
    T_star, x_lat = latency_optimal(sc)
    E_star, x_en, _ = energy_optimal(sc)
    T_en = float(objectives(x_en[None], sc)[0, 0])
    E_lat = float(objectives(x_lat[None], sc)[0, 1])
    dT, dE = T_en - T_star, E_lat - E_star
    lb0, ub0 = bounds_z(sc)
    s = np.append(ub0, T_en)
    lo, hi = np.append(lb0, T_star) / s, np.append(ub0, T_en) / s
    A = budget_jac(sc)
    cnt = Counter(limit=TOTAL)
    one = np.ones((M_, 1))

    def f(y):
        cnt.tick(y)
        v = y * s
        return (v[-1] - T_star) / dT + (model(v[:-1], sc)["E"] - E_star) / dE

    def g(y):
        cnt.tick(y)
        v = y * s
        return np.append(energy_grad(v[:-1], sc) * s[:-1] / dE, s[-1] / dT)

    def ineq(y):
        cnt.tick(y)
        v = y * s
        m = model(v[:-1], sc)
        return np.concatenate([(v[-1] - m["Tl"]) / T_en, (v[-1] - m["Tt"] - m["Tm"]) / T_en, budget_values(v[:-1], sc)])

    def ineq_jac(y):
        cnt.tick(y)
        v = y * s
        Jl, Jo = time_jacobians(v[:-1], sc)
        J = np.vstack([np.hstack([-Jl / T_en, one * 0 + 1 / T_en]), np.hstack([-Jo / T_en, one * 0 + 1 / T_en]),
                       np.hstack([A, np.zeros((2, 1))])])
        return J * s

    def y_of(zz):
        """Scaled NBI variables of a decision vector, with the epigraph variable at its smallest feasible value."""
        m = model(zz, sc)
        tau = float(np.clip(max(m["Tl"].max(), (m["Tt"] + m["Tm"]).max()), T_star, T_en))
        return np.clip(np.append(zz, tau) / s, lo, hi)

    def merit(q, c):
        """(violation, objective) of a scaled point, without counting evaluations (bookkeeping only)."""
        v = q * s
        m = model(v[:-1], sc)
        e = abs((v[-1] - T_star) / dT - (m["E"] - E_star) / dE - c)
        ti = max(0.0, float(max(m["Tl"].max(), (m["Tt"] + m["Tm"]).max()) - v[-1]) / T_en)
        bu = max(0.0, -float(np.min(budget_values(v[:-1], sc))))
        viol = e + ti + bu
        return (viol > 1e-6, viol if viol > 1e-6 else (v[-1] - T_star) / dT + (m["E"] - E_star) / dE)

    rng = np.random.default_rng(seed)
    z, stats, Z = from_genes(x_lat, sc), [], []
    y = np.append(z, T_star) / s
    next_act, gap, n_act = 1, 1, 0
    for k in range(1, n_b + 1):
        if cnt.n >= TOTAL:
            break
        c = 1 - 2 * (1 - k / (n_b + 1))

        def eq(q, c=c):
            cnt.tick(q)
            v = q * s
            return np.array([(v[-1] - T_star) / dT - (model(v[:-1], sc)["E"] - E_star) / dE - c])

        def eq_jac(q):
            cnt.tick(q)
            v = q * s
            return (np.append(-energy_grad(v[:-1], sc) / dE, 1 / dT) * s)[None]

        left = TOTAL - cnt.n
        allow = int(min(left, borrow * left / (n_b - k + 1)))
        cons = [{"type": "eq", "fun": eq, "jac": eq_jac}, {"type": "ineq", "fun": ineq, "jac": ineq_jac}]
        y, status = _slsqp_counted(f, g, cons, y, lo, hi, cnt, allow, per_iter=6)   # six counted calls per iteration
        y2 = None
        if restart == "activation" and k >= next_act:
            za = activated_start((y * s)[:-1], sc)
            if za is not None:
                y2 = y_of(za); n_act += 1
        elif restart == "random":
            y2 = y_of(random_z(sc, rng))
        if y2 is not None and cnt.n < TOTAL:
            left = TOTAL - cnt.n
            allow = int(min(left, borrow * left / (n_b - k + 1)))
            y2, status2 = _slsqp_counted(f, g, cons, y2, lo, hi, cnt, allow, per_iter=6)
            better = merit(y2, c) < merit(y, c)
            if better:
                y, status = y2, status2
            if restart == "activation":
                gap = 1 if better else 2 * gap
                next_act = k + gap
        stats.append(dict(beta=1 - k / (n_b + 1), status=status))
        Z.append((y * s)[:-1])
    F, X = _front_from_z(Z, sc, x_lat, x_en)
    return F, X, cnt.n, dict(levels=stats, n_activated=n_act)


def run_aec_sqp(sc, seed=0, n_solves=K_FRONT, borrow=2.0):
    """Adaptive epsilon-constraint method in the spirit of [Laumanns et al. 2006]: starting from the two exact anchors,
    each epsilon-constraint problem (the same problem, solver and gradients as ACE-SQP, without activation restart) is
    placed at the mid-latency of the largest gap between neighbouring points of the current front, measured in the
    objective space normalized by the anchors, and warm-started from the lower-latency end of that gap. A gap that
    yields no new non-dominated point is not tried again."""
    T_star, x_lat = latency_optimal(sc)
    E_star, x_en, _ = energy_optimal(sc)
    F_lat, F_en = objectives(x_lat[None], sc)[0], objectives(x_en[None], sc)[0]
    span = np.array([F_en[0] - T_star, F_lat[1] - E_star])
    E_ref = float(F_lat[1])
    cnt = Counter(limit=TOTAL)
    rep, prob = SlackRepair(), MECProblem(sc)
    pts = [(F_lat, from_genes(x_lat, sc)), (F_en, from_genes(x_en, sc))]    # (objectives, physical z)
    tried, stats, Z = set(), [], []
    for k in range(n_solves):
        if cnt.n >= TOTAL:
            break
        front = sorted(pts, key=lambda p: p[0][0])
        Fa = np.array([p[0] for p in front])
        keep = NonDominatedSorting().do(Fa, only_non_dominated_front=True)
        front = [front[i] for i in sorted(keep, key=lambda i: Fa[i, 0])]
        gaps = []
        for a, b in zip(front[:-1], front[1:]):
            key = (round(float(a[0][0]), 9), round(float(b[0][0]), 9))
            if key not in tried and b[0][0] - a[0][0] > 1e-9 * T_star:
                gaps.append((float(np.linalg.norm((b[0] - a[0]) / span)), key, a, b))
        if not gaps:
            break
        _, key, a, b = max(gaps, key=lambda t: t[0])
        tried.add(key)
        eps = 0.5 * (a[0][0] + b[0][0])
        left = TOTAL - cnt.n
        allow = int(min(left, borrow * left / (n_solves - k)))
        try:
            z, res, nit = solve_eps(a[1], eps * (1 + 1e-9), sc, cnt, allow, E_ref)
            status = int(res.status)
        except Exception:
            z, status = a[1], -1
        stats.append(dict(eps=eps, status=status))
        x = rep._do(prob, to_genes(z, sc)[None])[0]
        Fz = objectives(x[None], sc)[0]
        pts.append((Fz, z))
        Z.append(z)
    F, X = _front_from_z(Z, sc, x_lat, x_en)
    return F, X, cnt.n, dict(levels=stats)


class WSProblem(Problem):
    def __init__(self, sc, w, ideal, nadir):
        self.sc, self.w, self.ideal, self.span = sc, w, ideal, nadir - ideal
        base = MECProblem(sc)
        super().__init__(n_var=base.n_var, n_obj=1, xl=base.xl, xu=base.xu)

    def _evaluate(self, X, out, *a, **k):
        Fn = (objectives(X, self.sc) - self.ideal) / self.span
        out["F"] = self.w * Fn[:, 0] + (1 - self.w) * Fn[:, 1]


def run_ws_ga(sc, seed, n_w=10, ops=None):
    T_star, x_lat = latency_optimal(sc)
    E_star, _, _ = energy_optimal(sc)  # the same ideal point as WS-SQP
    ideal = np.array([T_star, E_star])
    nadir = np.array([t_nadir(sc), objectives(x_lat[None], sc)[0, 1]])
    pop, ngen = 50, TOTAL // (n_w * 50)
    rng = np.random.default_rng(seed)
    Fs = []
    for w in np.linspace(0, 1, n_w):
        res = minimize(WSProblem(sc, w, ideal, nadir),
                       GA(pop_size=pop, sampling=seed_population_exact(sc, pop, rng), **ops_kwargs(ops)),
                       ("n_gen", ngen), seed=seed, verbose=False)
        Fs.append(objectives(np.atleast_2d(res.X), sc)[0])
    return np.array(Fs)


# ---------------------------------------------------------------------------- dispatcher
STOCHASTIC = ["SR-NSGA-II+SQP", "SR-NSGA-II", "NSGA-II+S", "NSGA-II", "SPEA2+S", "MOEA/D+S", "WS-GA+S", "WS-SQP", "Random"]
DETERMINISTIC = [PROP, "BCD-eps", "NBI-SQP", "AEC-SQP"]          # one run per scenario
METHODS = [PROP, "BCD-eps", "NBI-SQP", "AEC-SQP"] + STOCHASTIC
# Extensions of ACE-SQP (v11): they start from the ACE-SQP front and are not independent competitors. They are excluded
# from the family of tests against ACE-SQP, from the Friedman ranks and from the reference front without ACE-SQP; the
# analysis reports instead whether they improve on ACE-SQP.
EXTENSIONS = ["SR-NSGA-II+SQP"]
# Diagnostic variants (v13): which ingredient makes the difference? Analyzed separately (analysis_v13.py).
#   ACE-SQP (random restart): the restart of ACE-SQP from a random point instead of the activated start (stochastic)
#   NBI-SQP+act: NBI-SQP with the activation restart of ACE-SQP (deterministic)
#   NBI-SQP+rand: NBI-SQP with one random start per subproblem (stochastic; about twice the run time of NBI-SQP)
VARIANTS = ["ACE-SQP (random restart)", "NBI-SQP+act", "NBI-SQP+rand"]
ABLATIONS = {"no activation": dict(activation=False), "no E* anchor": dict(use_energy_anchor=False),
             "no repair": dict(repair=False), "no continuation": dict(continuation=False),
             "cold start": dict(cold_seed=0), "finite differences": dict(use_grad=False),
             "with backward sweep": dict(backward=True), "K=12 levels": dict(K=12)}


def run_method(name, sc, seed, cfg=None):
    """cfg: settings selected on the tuning scenarios (see tuning.py)."""
    cfg = cfg or {}
    ops = cfg.get("ops", {}).get(name)
    tr, t0 = Trace(), time.time()
    extra = {}
    if name == PROP:
        c = cfg.get("ace", {})
        F, X, used, stats = run_ace(sc, seed, shape=c.get("shape", 2.0), borrow=c.get("borrow", 1.0))
        hist, extra = [], dict(evals=used, sqp_stats=stats)
    elif name == "SR-NSGA-II+SQP":
        c = cfg.get("hybrid", {})
        F, X, used, n_gen, stats, h = run_hybrid(sc, seed, K=c.get("K", 12), rho=c.get("rho", 0.45), ops=ops, record=True,
                                                 borrow=cfg.get("ace", {}).get("borrow", 1.0))
        hist, extra = [(n + used, f) for n, f in h], dict(evals=TOTAL, sqp_evals=used, n_gen=n_gen)
    elif name == "WS-SQP":
        F, used = run_ws_sqp(sc, seed)
        X, hist, extra = None, [], dict(evals=used)
    elif name == "BCD-eps":
        import bcd
        c = cfg.get("ace", {})
        F, X, used, st = bcd.run_bcd(sc, shape=c.get("shape", 2.0), borrow=c.get("borrow", 1.0))
        hist, extra = [], dict(evals=used, bcd_stats=st)
    elif name in ("NBI-SQP", "AEC-SQP"):
        c = cfg.get("ace", {})
        fn = run_nbi_sqp if name == "NBI-SQP" else run_aec_sqp
        F, X, used, st = fn(sc, seed, borrow=c.get("borrow", 2.0))
        hist, extra = [], dict(evals=used, sqp_stats=dict(levels=st["levels"], n_kept=len(st["levels"]), n_activated=0))
    elif name in VARIANTS:                     # v13 diagnostic variants (not part of the main comparison)
        c = cfg.get("ace", {})
        if name == "ACE-SQP (random restart)":
            F, X, used, st = run_ace(sc, seed, shape=c.get("shape", 2.0), borrow=c.get("borrow", 1.0),
                                     restart="random", restart_seed=seed)
        else:
            F, X, used, st = run_nbi_sqp(sc, seed, borrow=c.get("borrow", 2.0),
                                         restart="activation" if name == "NBI-SQP+act" else "random")
            st = dict(levels=st["levels"], n_kept=len(st["levels"]), n_activated=st["n_activated"])
        hist, extra = [], dict(evals=used, sqp_stats=st)
    elif name == "WS-GA+S":
        F, X, hist = run_ws_ga(sc, seed, ops=ops), None, []
    elif name == "Random":
        rng = np.random.default_rng(seed)
        X = rng.random((TOTAL, 4 * sc.M))
        X[:, sc.M:3 * sc.M] = np.maximum(X[:, sc.M:3 * sc.M], W_MIN)
        F, X, hist = objectives(X, sc), None, []
    else:
        seeded = seed_population_exact(sc, POP, np.random.default_rng(seed))
        if name == "NSGA-II":
            alg = NSGA2(pop_size=POP, **ops_kwargs(ops))
        elif name == "NSGA-II+S":
            alg = NSGA2(pop_size=POP, sampling=seeded, **ops_kwargs(ops))
        elif name == "SR-NSGA-II":
            alg = NSGA2(pop_size=POP, sampling=seeded, repair=SlackRepair(), **ops_kwargs(ops))
        elif name == "SPEA2+S":
            alg = SPEA2(pop_size=POP, sampling=seeded, **ops_kwargs(ops))
        elif name == "MOEA/D+S":
            md = cfg.get("moead", dict(n_neighbors=15, prob_neighbor_mating=0.7))
            alg = MOEAD(get_reference_directions("uniform", 2, n_partitions=POP - 1), sampling=seeded, **md)
        else:
            raise ValueError(name)
        res = minimize(MECProblem(sc), alg, ("n_gen", NGEN), seed=seed, callback=tr, verbose=False)
        F, X, hist = res.F, res.X, tr.hist
    return dict(F=nondom(F), X=X, hist=hist, time=time.time() - t0, **extra)

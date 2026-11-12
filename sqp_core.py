"""
SLSQP building blocks with closed-form gradients and scaled variables.

Physical vector z = [P | f (GHz) | F (GHz) | B (MHz)].
Accounting: every call of an objective, a constraint vector, an objective gradient or a
constraint Jacobian counts as ONE evaluation, because each of them costs one pass over the model.
With finite differences (use_grad=False) SciPy calls the counted objective/constraints itself.
"""
import numpy as np
from scipy.optimize import minimize as sp_minimize
from model import Scenario, W_MIN, decode

LN2 = np.log(2.0)
GHZ, MHZ = 1e9, 1e6
FLOOR = 1e-4     # lower bound on F_m and B_m, relative to their budgets (avoids a nearly singular rate)
FTOL = 1e-7      # on the energy scaled by the energy of the latency anchor
RESTARTS = 2     # SLSQP is restarted from its last iterate when it stops without convergence


class BudgetExceeded(Exception):
    """Raised by Counter.tick when the evaluation budget is used up (hard cap)."""


class Counter:
    def __init__(self, limit=None):
        self.n = 0
        self.limit = limit
        self.last = None          # last evaluated point, used when a solver is stopped by the cap

    def tick(self, y=None, k=1):
        if self.limit is not None and self.n + k > self.limit:
            raise BudgetExceeded
        self.n += k
        if y is not None:
            self.last = np.array(y, copy=True)


def bounds_z(sc: Scenario, floor=FLOOR):
    M = sc.M
    lb = np.concatenate([np.zeros(M), np.full(M, sc.f_min) / GHZ,
                         np.full(M, floor * sc.F_max) / GHZ, np.full(M, floor * sc.B_max) / MHZ])
    ub = np.concatenate([np.ones(M), sc.f_maxm / GHZ, np.full(M, sc.F_max) / GHZ, np.full(M, sc.B_max) / MHZ])
    return lb, ub


def split(z, M):
    return z[:M], z[M:2 * M] * GHZ, z[2 * M:3 * M] * GHZ, z[3 * M:4 * M] * MHZ


def model(z, sc):
    """Times, energy and the partial derivatives needed by the gradients."""
    M = sc.M
    P, f, F, B = split(z, M)
    k = sc.gamma * sc.B_ref
    R = B * np.log2(1 + k / B)
    dR = np.log2(1 + k / B) - (k / B) / ((1 + k / B) * LN2)
    Tl = (1 - P) * sc.C / f
    Tt = P * sc.bits / R
    Tm = P * sc.C / F
    E = np.sum(sc.kappa * f ** 2 * (1 - P) * sc.C + sc.p_tx * Tt)
    return dict(P=P, f=f, F=F, B=B, R=R, dR=dR, Tl=Tl, Tt=Tt, Tm=Tm, E=E)


def energy_grad(z, sc):
    M = sc.M
    m = model(z, sc)
    g = np.zeros(4 * M)
    g[:M] = -sc.kappa * m["f"] ** 2 * sc.C + sc.p_tx * sc.bits / m["R"]
    g[M:2 * M] = 2 * sc.kappa * m["f"] * (1 - m["P"]) * sc.C * GHZ
    g[3 * M:4 * M] = -sc.p_tx * m["P"] * sc.bits * m["dR"] / m["R"] ** 2 * MHZ
    return g


def time_jacobians(z, sc):
    """d(Tl)/dz and d(Tt+Tm)/dz, each of shape (M, 4M)."""
    M = sc.M
    m = model(z, sc)
    i = np.arange(M)
    Jl, Jo = np.zeros((M, 4 * M)), np.zeros((M, 4 * M))
    Jl[i, i] = -sc.C / m["f"]
    Jl[i, M + i] = -(1 - m["P"]) * sc.C / m["f"] ** 2 * GHZ
    Jo[i, i] = sc.bits / m["R"] + sc.C / m["F"]
    Jo[i, 2 * M + i] = -m["P"] * sc.C / m["F"] ** 2 * GHZ
    Jo[i, 3 * M + i] = -m["P"] * sc.bits * m["dR"] / m["R"] ** 2 * MHZ
    return Jl, Jo


def budget_jac(sc):
    """Jacobian of the scaled budgets 1 - sum(F)/F_max and 1 - sum(B)/B_max with respect to z."""
    M = sc.M
    A = np.zeros((2, 4 * M))
    A[0, 2 * M:3 * M] = -GHZ / sc.F_max
    A[1, 3 * M:4 * M] = -MHZ / sc.B_max
    return A


def budget_values(z, sc):
    M = sc.M
    return np.array([1 - z[2 * M:3 * M].sum() * GHZ / sc.F_max, 1 - z[3 * M:4 * M].sum() * MHZ / sc.B_max])


def to_genes(z, sc):
    M = sc.M
    P, f, F, B = split(z, M)
    u = (np.clip(f, sc.f_min, sc.f_maxm) - sc.f_min) / (sc.f_maxm - sc.f_min)
    return np.concatenate([np.clip(P, 0, 1), np.maximum(F / F.max(), W_MIN), np.maximum(B / B.max(), W_MIN), u])


def from_genes(x, sc):
    P, F, B, f = (a[0] for a in decode(x[None], sc))
    lb, ub = bounds_z(sc)
    return np.clip(np.concatenate([P, f / GHZ, F / GHZ, B / MHZ]), lb, ub)


def solve_eps(z0, eps, sc, cnt, max_evals, E_ref, use_grad=True, ftol=FTOL, restarts=RESTARTS):
    """min E/E_ref s.t. all completion times <= eps, in variables scaled by their upper bounds.
    Returns z, the last SLSQP result and the number of SLSQP iterations."""
    lb, ub = bounds_z(sc)
    s = ub
    A = budget_jac(sc)

    def obj(y):
        cnt.tick(y)
        return model(y * s, sc)["E"] / E_ref

    def grad(y):
        cnt.tick(y)
        return energy_grad(y * s, sc) * s / E_ref

    def con(y):
        cnt.tick(y)
        z = y * s
        m = model(z, sc)
        return np.concatenate([1 - m["Tl"] / eps, 1 - (m["Tt"] + m["Tm"]) / eps, budget_values(z, sc)])

    def jac(y):
        cnt.tick(y)
        Jl, Jo = time_jacobians(y * s, sc)
        return np.vstack([-Jl * s / eps, -Jo * s / eps, A * s])

    y = np.clip(z0, lb, ub) / s
    bnds = list(zip(lb / s, ub / s))
    cons = {"type": "ineq", "fun": con}
    if use_grad:
        cons["jac"] = jac
    # one iteration costs about four calls with gradients, and 2(n+1) calls with finite differences;
    # the first attempt may use the whole allowance, a restart only what is left of it
    per_iter = 4 if use_grad else 2 * (lb.size + 1)
    res, nit, n0 = None, 0, cnt.n
    for _ in range(restarts + 1):
        left = max_evals - (cnt.n - n0)
        if left < per_iter:
            break
        try:
            res = sp_minimize(obj, y, jac=grad if use_grad else None, method="SLSQP", bounds=bnds, constraints=[cons],
                              options=dict(maxiter=max(1, left // per_iter), ftol=ftol))
        except BudgetExceeded:                     # hard cap: keep the last evaluated point
            if cnt.last is not None:
                y = np.clip(cnt.last, lb / s, ub / s)
            class _R: status, nit = 9, 0
            res = _R()
            break
        y = np.clip(res.x, lb / s, ub / s)
        nit += res.nit
        if res.status == 0:
            break
    if res is None:
        class _R: status, nit = 9, 0
        res = _R()
    return y * s, res, nit


def eps_levels(T_star, T_max, K, shape=2.0):
    return T_star + (T_max - T_star) * np.linspace(0, 1, K) ** shape

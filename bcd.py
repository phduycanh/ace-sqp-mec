"""
BCD-eps: block coordinate descent baseline on the same ladder of epsilon-constraint problems as ACE-SQP.
This is the decomposition commonly used for energy-latency resource allocation in MEC:
  block 1 (P, f | F, B): separable over the UDs. For a given P the best frequency is f = max(f_min, (1-P)C/eps)
          (Proposition 3), so each UD minimizes the convex one-dimensional function
              g(P) = kappa C (1-P) max(f_min^2, ((1-P)C/eps)^2) + p_tx P D / R(B)
          over [max(0, 1 - eps f_max/C), min(1, eps / (D/R(B) + C/F))]  (vectorized golden-section search);
  block 2 (F, B | P): the convex problem  min sum p_tx P D / R(B)  s.t.  P D/R(B) + P C/F <= eps, budgets, floors,
          solved by SLSQP with closed-form gradients (convex, hence globally).
Blocks alternate until the energy changes by less than 1e-6 (relative) or the allowance of the level is spent.
Level k is warm-started from level k-1, starting from the latency anchor; the same bookkeeping as ACE-SQP
(best feasible point per level, monotone safeguard, slack repair, both anchors added) is used.
Every golden-section step, objective, constraint, gradient or Jacobian call counts as one evaluation.
"""
import numpy as np
from scipy.optimize import minimize as sp_minimize
from model import objectives, SlackRepair, MECProblem
from model_ext import energy_optimal
from sqp_core import Counter, BudgetExceeded, bounds_z, model, to_genes, from_genes, eps_levels, FTOL, GHZ, MHZ

GOLD = (np.sqrt(5) - 1) / 2


def p_step(z, sc, eps, cnt, it=40):
    M = sc.M
    P, f, F, B = z[:M].copy(), z[M:2 * M] * GHZ, z[2 * M:3 * M] * GHZ, z[3 * M:] * MHZ
    R = B * np.log2(1 + sc.gamma * sc.B_ref / B)
    lo = np.maximum(0.0, 1 - eps * sc.f_maxm / sc.C)
    hi = np.minimum(1.0, eps / (sc.bits / R + sc.C / F))
    ok = lo <= hi
    lo, hi = np.where(ok, lo, P), np.where(ok, hi, P)        # infeasible box: keep the current P

    def g(p):
        q = 1 - p
        return sc.kappa * sc.C * q * np.maximum(sc.f_min ** 2, (q * sc.C / eps) ** 2) + sc.p_tx * p * sc.bits / R

    a, b = lo.copy(), hi.copy()
    c, d = b - GOLD * (b - a), a + GOLD * (b - a)
    gc, gd = g(c), g(d)
    for _ in range(it):
        cnt.tick(k=2)                   # both interior points are re-evaluated (vectorized over the UDs)
        left = gc < gd
        b = np.where(left, d, b); a = np.where(left, a, c)
        d_new = np.where(left, c, a + GOLD * (b - a)); c_new = np.where(left, b - GOLD * (b - a), d)
        c, d = c_new, d_new
        gc, gd = g(c), g(d)
    Pn = 0.5 * (a + b)
    for x in (lo, hi):                                          # the minimum may sit at an end of the box
        Pn = np.where(g(x) < g(Pn), x, Pn)
    fn = np.clip(np.maximum(sc.f_min, (1 - Pn) * sc.C / eps), sc.f_min, sc.f_maxm)
    return np.concatenate([Pn, fn / GHZ, z[2 * M:3 * M], z[3 * M:]])


def fb_step(z, sc, eps, cnt, max_evals):
    M = sc.M
    lb, ub = bounds_z(sc)
    P = z[:M]
    s = ub[2 * M:]
    k = sc.gamma * sc.B_ref
    E_ref = max(1e-12, float(np.sum(sc.p_tx * P * sc.bits / (z[3 * M:] * MHZ * np.log2(1 + k / (z[3 * M:] * MHZ))))) + 1e-3)

    def parts(y):
        F, B = y[:M] * s[:M] * GHZ, y[M:] * s[M:] * MHZ
        R = B * np.log2(1 + k / B)
        dR = np.log2(1 + k / B) - (k / B) / ((1 + k / B) * np.log(2))
        return F, B, R, dR

    def obj(y):
        cnt.tick(y)
        F, B, R, _ = parts(y)
        return float(np.sum(sc.p_tx * P * sc.bits / R)) / E_ref

    def grad(y):
        cnt.tick(y)
        F, B, R, dR = parts(y)
        g = np.zeros(2 * M)
        g[M:] = -sc.p_tx * P * sc.bits * dR / R ** 2 * MHZ * s[M:] / E_ref
        return g

    def con(y):
        cnt.tick(y)
        F, B, R, _ = parts(y)
        return np.concatenate([1 - (P * sc.bits / R + P * sc.C / F) / eps,
                               [1 - F.sum() / sc.F_max, 1 - B.sum() / sc.B_max]])

    def jac(y):
        cnt.tick(y)
        F, B, R, dR = parts(y)
        J = np.zeros((M + 2, 2 * M))
        i = np.arange(M)
        J[i, i] = P * sc.C / F ** 2 / eps * GHZ * s[:M]
        J[i, M + i] = P * sc.bits * dR / R ** 2 / eps * MHZ * s[M:]
        J[M, :M] = -GHZ * s[:M] / sc.F_max
        J[M + 1, M:] = -MHZ * s[M:] / sc.B_max
        return J

    y0 = z[2 * M:] / s
    bnds = list(zip(lb[2 * M:] / s, ub[2 * M:] / s))
    if max_evals < 8:
        return z
    try:
        res = sp_minimize(obj, y0, jac=grad, method="SLSQP", bounds=bnds,
                          constraints=[{"type": "ineq", "fun": con, "jac": jac}],
                          options=dict(maxiter=max(1, max_evals // 4), ftol=FTOL))
        y = res.x
    except BudgetExceeded:                      # hard cap: keep the last evaluated point
        y = cnt.last if cnt.last is not None and cnt.last.shape == y0.shape else y0
    y = np.clip(y, lb[2 * M:] / s, ub[2 * M:] / s)
    return np.concatenate([z[:2 * M], y * s])


def feasible(z, sc, eps):
    m = model(z, sc)
    return max(m["Tl"].max(), (m["Tt"] + m["Tm"]).max()) <= eps * (1 + 1e-6), float(m["E"])


def run_bcd(sc, budget=30000, K=98, shape=2.0, borrow=1.0, tol=1e-6):
    from model import latency_optimal
    T_star, x_lat = latency_optimal(sc)
    levels = eps_levels(T_star, float(np.max(sc.C / sc.f_min)), K, shape)
    cnt = Counter(limit=budget)                  # hard cap on the evaluations
    z = from_genes(x_lat, sc)
    best, n_iter = [None] * K, []
    for i, eps in enumerate(levels):
        left = budget - cnt.n
        if left <= 0:
            break
        allow = min(left, borrow * left / (K - i))
        n0, E_prev, it = cnt.n, np.inf, 0
        while cnt.n - n0 < allow:
            it += 1
            try:
                z1 = p_step(z, sc, eps, cnt)
            except BudgetExceeded:
                break
            z2 = fb_step(z1, sc, eps, cnt, int(allow - (cnt.n - n0)))
            ok, E = feasible(z2, sc, eps)
            if ok:
                z = z2
            else:
                ok1, _ = feasible(z1, sc, eps)
                if ok1:
                    z = z1
                break
            if abs(E_prev - E) <= tol * max(E, 1e-12):
                break
            E_prev = E
        n_iter.append(it)
        ok, E = feasible(z, sc, eps)
        if ok:
            best[i] = (E, z.copy())
        if i > 0 and best[i - 1] is not None and (best[i] is None or best[i - 1][0] < best[i][0]):
            best[i] = best[i - 1]
            z = best[i][1].copy()
    Z = [b[1] for b in best if b is not None]
    X = np.array([to_genes(q, sc) for q in Z])
    X = SlackRepair()._do(MECProblem(sc), X)
    X = np.vstack([x_lat[None], X, energy_optimal(sc)[1][None]])
    return objectives(X, sc), X, cnt.n, dict(levels=len(Z), iters=float(np.mean(n_iter)) if n_iter else 0.0)

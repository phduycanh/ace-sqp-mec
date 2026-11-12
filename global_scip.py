"""
Global solution of the epsilon-constraint problem (eq. (9) of the paper) with SCIP (spatial branch-and-bound).
Exactly the problem that ACE-SQP solves locally: same floors delta_F = 1e-4 F_max, delta_B = 1e-4 B_max.
Units: P, f [GHz], F [GHz], B [MHz], times [s], energy [J].
  T_loc:  (1-P) C <= eps f                        (linear)
  T_off:  t1 + t2 <= eps,  t1 r >= P D,  t2 F >= P C
  rate:   r ln2 <= B log(B + k) - B log(B)         (hypograph of the concave rate)
  energy: e >= sum kappa' f^2 (1-P) C + p_tx t1
Returns the best primal value, the dual (lower) bound and the solver status.
"""
import numpy as np
from pyscipopt import Model, log as slog
from model import Scenario
from sqp_core import FLOOR

LN2 = np.log(2.0)


def solve_global(sc: Scenario, eps, time_limit=600.0, gap=1e-4, threads=1, verbose=False):
    M = sc.M
    C = sc.C / 1e9                      # Gcycles
    D = sc.bits / 1e6                   # Mbit
    k = sc.gamma * sc.B_ref / 1e6       # MHz
    Fm, Bm = sc.F_max / 1e9, sc.B_max / 1e6
    kap = sc.kappa * 1e27               # kappa (Hz)^2 C(cycles) = kap f(GHz)^2 C(Gcycles)
    m = Model()
    if not verbose:
        m.hideOutput()
    m.setParam("limits/time", time_limit)
    m.setParam("limits/gap", gap)
    m.setParam("parallel/maxnthreads", threads)
    fmin, fmx = sc.f_min / 1e9, sc.f_maxm / 1e9
    # the local branch forces P >= 1 - eps f_max / C; for a given P the best frequency is
    # max(f_min, (1-P) C / eps) (Proposition 3), so f is eliminated and the local energy
    # kap C q max(f_min^2, (q C / eps)^2), q = 1 - P, is convex
    P = [m.addVar(f"P{i}", lb=max(0.0, 1 - eps * fmx[i] / C[i]), ub=1) for i in range(M)]
    L = [m.addVar(f"L{i}", lb=0) for i in range(M)]
    F = [m.addVar(f"F{i}", lb=FLOOR * Fm, ub=Fm) for i in range(M)]
    B = [m.addVar(f"B{i}", lb=FLOOR * Bm, ub=Bm) for i in range(M)]
    r = [m.addVar(f"r{i}", lb=0, ub=k[i] / LN2) for i in range(M)]
    t1 = [m.addVar(f"t1_{i}", lb=0, ub=eps) for i in range(M)]
    t2 = [m.addVar(f"t2_{i}", lb=0, ub=eps) for i in range(M)]
    emax = float(np.sum(kap * (sc.f_maxm / 1e9) ** 2 * C) + sc.p_tx * M * eps)
    e = m.addVar("e", lb=0, ub=emax)
    m.addCons(sum(F) <= Fm)
    m.addCons(sum(B) <= Bm)
    for i in range(M):
        m.addCons(L[i] >= kap * C[i] * fmin ** 2 * (1 - P[i]))
        m.addCons(L[i] >= kap * C[i] ** 3 / eps ** 2 * (1 - P[i]) ** 3)
        m.addCons(t1[i] + t2[i] <= eps)
        m.addCons(t1[i] * r[i] >= P[i] * D[i])
        m.addCons(t2[i] * F[i] >= P[i] * C[i])
        m.addCons(r[i] * LN2 <= B[i] * slog(B[i] + k[i]) - B[i] * slog(B[i]))
    m.addCons(e >= sum(L[i] + sc.p_tx * t1[i] for i in range(M)))
    m.setObjective(e, "minimize")
    m.optimize()
    status = m.getStatus()
    primal = m.getObjVal() if m.getNSols() > 0 else np.inf
    return dict(primal=float(primal), dual=float(m.getDualbound()), status=status, time=float(m.getSolvingTime()),
                nodes=int(m.getNNodes()))


if __name__ == "__main__":
    import time
    import methods as Mt
    from sqp_core import eps_levels
    for MM in (3, 5):
        sc = Scenario(M=MM, seed=8001)
        T = Mt.latency_optimal(sc)[0]
        lev = eps_levels(T, Mt.t_nadir(sc), Mt.K_FRONT, 2.0)
        F = Mt.run_ace(sc)[0]
        for kk in (5, 20, 50, 80):
            g = solve_global(sc, lev[kk], time_limit=120)
            ace = F[F[:, 0] <= lev[kk] * (1 + 1e-6), 1].min()
            print(MM, kk, round(lev[kk], 3), "ACE", round(ace, 5), "SCIP", round(g["primal"], 5), "LB", round(g["dual"], 5),
                  g["status"], round(g["time"], 1), "s", flush=True)

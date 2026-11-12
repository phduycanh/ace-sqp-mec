"""
V13 (reviewer item W3 of the critique of main_v13): how far is the latency-only GA of Elkawkagy et al. [9] from the
exact latency optimum T*? GA of pymoo minimizing J1 only, random initial population (no anchors, as in [9]),
population 100, 300 generations (the common budget of 30 000 evaluations), operators of WS-GA+S selected on the tuning
scenarios; the 30 E1 scenarios, five runs each (seeds 0-4). The local frequency is a decision variable here, but
minimizing J1 drives it to its ceiling, as fixed in [9].
Output: results/ga9_latency.json (gap of the best J1 to T*, in %).
"""
import json, os, time
import numpy as np
from multiprocessing import Pool
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
import methods as M
from model import Scenario, objectives
import experiments_rev2 as XR2

CFG = json.load(open(os.path.join(M.RESULTS, "config.json")))


class LatencyProblem(Problem):
    def __init__(self, sc):
        self.sc = sc
        base = M.MECProblem(sc)
        super().__init__(n_var=base.n_var, n_obj=1, xl=base.xl, xu=base.xu)

    def _evaluate(self, X, out, *a, **k):
        out["F"] = objectives(X, self.sc)[:, 0]


def job(args):
    s, d = args
    M.init_worker()
    sc = Scenario(M=10, seed=s)
    T = M.latency_optimal(sc)[0]
    ops = CFG.get("ops", {}).get("WS-GA+S")
    res = minimize(LatencyProblem(sc), GA(pop_size=M.POP, **M.ops_kwargs(ops)), ("n_gen", M.NGEN), seed=d, verbose=False)
    return dict(s=s, d=d, T=float(T), J1=float(np.min(res.F)), gap=float(100 * (np.min(res.F) / T - 1)))


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    t0 = time.time()
    jobs = [(s, d) for s in XR2.SCEN_E1 for d in range(5)]
    with Pool(10) as p:
        out = p.map(job, jobs)
    g = np.array([o["gap"] for o in out])
    per = {}
    for o in out:
        per.setdefault(o["s"], []).append(o["gap"])
    sm = np.array([np.mean(v) for v in per.values()])
    summ = dict(runs=out, mean=float(g.mean()), median=float(np.median(g)), min=float(g.min()), max=float(g.max()),
                scen_mean_min=float(sm.min()), scen_mean_max=float(sm.max()), n=len(out))
    json.dump(summ, open(os.path.join(M.RESULTS, "ga9_latency.json"), "w"), indent=1)
    print({k: v for k, v in summ.items() if k != "runs"}, round(time.time() - t0), "s")

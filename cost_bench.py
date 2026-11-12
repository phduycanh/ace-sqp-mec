"""Wall-clock cost of one counted evaluation: SLSQP callbacks (objective, constraints, gradient, Jacobian)
against the evaluation of one solution inside a population of 100 (as in the MOEAs) and alone.
Output: results/cost_bench.json"""
import json, os, time
import numpy as np
import methods as M
from model import Scenario, objectives, W_MIN
from sqp_core import bounds_z, model, energy_grad, time_jacobians, budget_jac, budget_values

def per_call(fn, n):
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1e6      # microseconds

res = {}
for m in (10, 20, 30, 50):
    sc = Scenario(M=m, seed=5001)
    lb, ub = bounds_z(sc)
    z = lb + np.random.default_rng(0).random(lb.size) * (ub - lb)
    s, A, eps = ub, budget_jac(sc), 10.0
    obj = lambda: model(z, sc)["E"]
    def con():
        mm = model(z, sc)
        return np.concatenate([1 - mm["Tl"] / eps, 1 - (mm["Tt"] + mm["Tm"]) / eps, budget_values(z, sc)])
    grad = lambda: energy_grad(z, sc) * s
    def jac():
        Jl, Jo = time_jacobians(z, sc)
        return np.vstack([-Jl * s / eps, -Jo * s / eps, A * s])
    X = np.random.default_rng(1).random((100, 4 * m)); X[:, m:3 * m] = np.maximum(X[:, m:3 * m], W_MIN)
    n = 20000
    r = dict(objective=per_call(obj, n), constraints=per_call(con, n), gradient=per_call(grad, n), jacobian=per_call(jac, n),
             moea_in_pop100=per_call(lambda: objectives(X, sc), n // 20) / 100, single=per_call(lambda: objectives(X[:1], sc), n))
    res[m] = r
    print(m, {k: round(v, 2) for k, v in r.items()}, flush=True)
json.dump(res, open(os.path.join(M.RESULTS, "cost_bench.json"), "w"), indent=1)

"""
Sequential wall-clock timing on an otherwise idle machine.
- Exact anchors with empty caches (fast implementation of model_ext): 3 scenarios per size.
- Methods with the anchors already cached (they are a one-off cost per scenario, reported separately and added
  in the manuscript to the methods that use them):
  every size: 3 development scenarios x 2 runs; M = 10: every method; M = 20, 30, 50: ACE-SQP, BCD-eps, hybrid,
  SR-NSGA-II and WS-SQP. Run sequentially on an idle machine on mains power, one BLAS thread (OMP_NUM_THREADS=1).
Medians are reported because a laptop that sleeps or throttles produces isolated outliers.
Output: results/timing.json
"""
import json, os, platform, time
import numpy as np
import model, model_ext
import methods as M
from model import Scenario

CFG = json.load(open(os.path.join(M.RESULTS, "config.json")))
SETS = {m: [5001, 5002, 5003] for m in (10, 20, 30, 50)}       # the same design for every size
REPS = 2
LARGE = [M.PROP, "BCD-eps", "SR-NSGA-II+SQP", "SR-NSGA-II", "WS-SQP"]          # methods timed for M >= 20
res = {"machine": platform.processor(), "python": platform.python_version(), "anchors": {}, "methods": {}}

for m, seeds in SETS.items():
    for s in seeds:
        sc = Scenario(M=m, seed=s)
        model._CACHE.clear(); model_ext._E_CACHE.clear()
        t0 = time.perf_counter(); model.latency_optimal(sc); t1 = time.perf_counter(); model_ext.energy_optimal(sc); t2 = time.perf_counter()
        res["anchors"][f"M{m}_{s}"] = dict(M=m, latency=t1 - t0, energy=t2 - t1)
        print("anchors", m, s, round(t1 - t0, 2), round(t2 - t1, 2), flush=True)

M.init_worker()   # every method below finds the anchors in the cache
for m, seeds in SETS.items():
    names = M.METHODS if m == 10 else LARGE
    reps = REPS
    for n in names:
        ts = []
        for s in seeds:
            sc = Scenario(M=m, seed=s)
            M.latency_optimal(sc); M.energy_optimal(sc)
            for d in range(reps):
                t0 = time.perf_counter()
                M.run_method(n, sc, d, CFG)
                ts.append(time.perf_counter() - t0)
        res["methods"][f"M{m}|{n}"] = dict(median=float(np.median(ts)), mean=float(np.mean(ts)), runs=[float(x) for x in ts])
        print(m, n, round(float(np.median(ts)), 2), flush=True)
for m in SETS:
    a = [v for v in res["anchors"].values() if v["M"] == m]
    res[f"anchor_median_M{m}"] = dict(latency=float(np.median([x["latency"] for x in a])), energy=float(np.median([x["energy"] for x in a])))
json.dump(res, open(os.path.join(M.RESULTS, "timing.json"), "w"), indent=1)
print(json.dumps(res, indent=1))

"""
V13: sequential wall-clock timing of the diagnostic variants (same design as timing.py / timing_v12.py: development
scenarios 5001-5003, two runs each, anchors cached, one BLAS thread, power throttling disabled). The run times stored
in raw_v13_*.pkl were measured in a parallel pool and are not comparable with those of the main runs; this script
times ACE-SQP, ACE-SQP (random restart), NBI-SQP and NBI-SQP+act in the same session.
Output: results/timing_v13.json (results/timing.json is not touched).
"""
import json, os, time
import numpy as np
import highqos
import methods as M
from model import Scenario

os.environ["OMP_NUM_THREADS"] = "1"
CFG = json.load(open(os.path.join(M.RESULTS, "config.json")))
P = os.path.join(M.RESULTS, "timing_v13.json")
print("power throttling disabled:", highqos.enable(), flush=True)
M.init_worker()
NAMES = [M.PROP, "ACE-SQP (random restart)", "NBI-SQP", "NBI-SQP+act"]
res = {}
for m in (10, 20, 30, 50):
    for s in (5001, 5002, 5003):
        sc = Scenario(M=m, seed=s)
        M.latency_optimal(sc); M.energy_optimal(sc)
    for n in NAMES:
        ts, ev = [], []
        for s in (5001, 5002, 5003):
            sc = Scenario(M=m, seed=s)
            for d in range(2):
                t0 = time.perf_counter(); r = M.run_method(n, sc, d, CFG); ts.append(time.perf_counter() - t0)
                ev.append(r.get("evals") or M.TOTAL)
        res[f"M{m}|{n}"] = dict(median=float(np.median(ts)), mean=float(np.mean(ts)), runs=[float(x) for x in ts],
                                evals=float(np.mean(ev)))
        print(m, n, round(float(np.median(ts)), 2), int(np.mean(ev)), flush=True)
        json.dump(res, open(P, "w"), indent=1)
print("ALL DONE", flush=True)

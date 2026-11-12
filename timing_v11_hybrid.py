"""
V11: sequential wall-clock timing of the hybrid with its external archive (ACE-SQP + evolutionary refinement), with the
same design as timing.py (development scenarios 5001-5003, two runs each, anchors cached, one BLAS thread, power
throttling disabled). Only the hybrid entries of results/timing.json are replaced; the v10 file is kept as
results/timing_v10.json.
"""
import json, os, shutil, time
import numpy as np
import highqos
import methods as M
from model import Scenario

CFG = json.load(open(os.path.join(M.RESULTS, "config.json")))
P = os.path.join(M.RESULTS, "timing.json")
if not os.path.exists(os.path.join(M.RESULTS, "timing_v10.json")):
    shutil.copy2(P, os.path.join(M.RESULTS, "timing_v10.json"))
res = json.load(open(os.path.join(M.RESULTS, "timing_v10.json")))
print("power throttling disabled:", highqos.enable(), flush=True)
M.init_worker()
for m in (10, 20, 30, 50):
    ts = []
    for s in (5001, 5002, 5003):
        sc = Scenario(M=m, seed=s)
        M.latency_optimal(sc); M.energy_optimal(sc)
        for d in range(2):
            t0 = time.perf_counter()
            M.run_method("SR-NSGA-II+SQP", sc, d, CFG)
            ts.append(time.perf_counter() - t0)
    res["methods"][f"M{m}|SR-NSGA-II+SQP"] = dict(median=float(np.median(ts)), mean=float(np.mean(ts)), runs=[float(x) for x in ts])
    print(m, round(float(np.median(ts)), 2), [round(x, 2) for x in ts], flush=True)
res["v11_note"] = "SR-NSGA-II+SQP entries re-timed with the external archive (timing_v11_hybrid.py)"
json.dump(res, open(P, "w"), indent=1)

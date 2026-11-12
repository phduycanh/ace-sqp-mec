"""
V12: sequential wall-clock timing (same design as timing.py: development scenarios 5001-5003, two runs each, anchors
cached, one BLAS thread, power throttling disabled) of
  - NBI-SQP and AEC-SQP for M = 10, 20, 30, 50 (added to results/timing.json; the v11 file is kept as timing_v11.json);
  - a single epsilon-constraint problem at (1 + delta) T*, delta in {0.10, 0.25}, started from the latency anchor,
    with the activation restart (reviewer item M9)                  -> results/timing.json, key "single".
"""
import json, os, shutil, time
import numpy as np
import highqos
import methods as M
from model import Scenario, objectives
from sqp_core import Counter, from_genes, model, solve_eps

CFG = json.load(open(os.path.join(M.RESULTS, "config.json")))
P = os.path.join(M.RESULTS, "timing.json")
if not os.path.exists(os.path.join(M.RESULTS, "timing_v11.json")):
    shutil.copy2(P, os.path.join(M.RESULTS, "timing_v11.json"))
res = json.load(open(os.path.join(M.RESULTS, "timing_v11.json")))
print("power throttling disabled:", highqos.enable(), flush=True)
M.init_worker()


def single(sc, d):
    T, x_lat = M.latency_optimal(sc)
    E0 = float(objectives(x_lat[None], sc)[0, 1])
    eps = T * (1 + d) * (1 + 1e-9)
    cnt = Counter(limit=M.TOTAL)
    z, _, _ = solve_eps(from_genes(x_lat, sc), eps, sc, cnt, M.TOTAL, E0)
    za = M.activated_start(z, sc)
    if za is not None:
        solve_eps(za, eps, sc, cnt, M.TOTAL - cnt.n, E0)
    return cnt.n


res["single"] = {}
for m in (10, 20, 30, 50):
    for n in ("NBI-SQP", "AEC-SQP"):
        ts = []
        for s in (5001, 5002, 5003):
            sc = Scenario(M=m, seed=s)
            M.latency_optimal(sc); M.energy_optimal(sc)
            for d in range(2):
                t0 = time.perf_counter(); M.run_method(n, sc, d, CFG); ts.append(time.perf_counter() - t0)
        res["methods"][f"M{m}|{n}"] = dict(median=float(np.median(ts)), mean=float(np.mean(ts)), runs=[float(x) for x in ts])
        print(m, n, round(float(np.median(ts)), 2), flush=True)
    for d in (0.10, 0.25):
        ts, ev = [], []
        for s in (5001, 5002, 5003):
            sc = Scenario(M=m, seed=s)
            for _ in range(2):
                t0 = time.perf_counter(); ev.append(single(sc, d)); ts.append(time.perf_counter() - t0)
        res["single"][f"M{m}|{d}"] = dict(median=float(np.median(ts)), runs=[float(x) for x in ts], evals=float(np.mean(ev)))
        print(m, "single", d, round(float(np.median(ts)), 3), flush=True)
res["v12_note"] = "NBI-SQP, AEC-SQP and the single-point solve timed by timing_v12.py"
json.dump(res, open(P, "w"), indent=1)

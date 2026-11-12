"""Check that the fast anchor implementation reproduces the cached anchors and time both versions.
Output: results/fast_anchors.json"""
import json, os, time
import numpy as np
import model, model_ext
import methods as M
from model import Scenario, objectives

cache = M.load_cache()
rows = []
for k, v in cache.items():
    d = dict(k)
    sc = Scenario(**d)
    model._CACHE.clear(); model_ext._E_CACHE.clear()
    t0 = time.perf_counter(); T, x = model_ext.latency_optimal_fast(sc); t1 = time.perf_counter()
    E, xe, info = model_ext.energy_optimal(sc); t2 = time.perf_counter()
    F_lat = objectives(x[None], sc)[0]
    rows.append(dict(key=str(d), M=d["M"], dT=abs(T - v["T"]) / v["T"], dE=abs(E - v["E"]) / v["E"],
                     same_set=info["offload_set"] == v["info"]["offload_set"], feas=bool(F_lat[0] <= T * (1 + 1e-6)),
                     t_lat_fast=t1 - t0, t_en_fast=t2 - t1, t_lat_old=v["t_lat"], t_en_old=v["t_en"]))
out = dict(n=len(rows), max_dT=max(r["dT"] for r in rows), max_dE=max(r["dE"] for r in rows),
           all_same_set=all(r["same_set"] for r in rows), all_feasible=all(r["feas"] for r in rows))
for m in sorted({r["M"] for r in rows}):
    g = [r for r in rows if r["M"] == m]
    out[f"M{m}"] = {q: float(np.median([r[q] for r in g])) for q in ("t_lat_fast", "t_en_fast", "t_lat_old", "t_en_old")}
json.dump(out, open(os.path.join(M.RESULTS, "fast_anchors.json"), "w"), indent=1)
print(json.dumps(out, indent=1))

"""Additional statistics quoted in the manuscript. Output: results/paper_stats.json"""
import json, os
import numpy as np
import pandas as pd
import analysis as A
import methods as M
from model import Scenario, objectives

M.init_worker()
out = A.load("e1")
st = {}
scen = sorted({o["kw"]["seed"] for o in out})

# ACE-SQP vs WS-SQP at single latency budgets
for p in (10, 25):
    a, w = [], []
    for s in scen:
        sc = Scenario(M=10, seed=s)
        T, x = M.latency_optimal(sc)
        E0 = objectives(x[None], sc)[0, 1]
        sv = lambda n: np.mean([100 * (1 - o["F"][o["F"][:, 0] <= T * (1 + p / 100), 1].min() / E0)
                                for o in out if o["name"] == n and o["kw"]["seed"] == s])
        a.append(sv(M.PROP)); w.append(sv("WS-SQP"))
    a, w = np.array(a), np.array(w)
    st[f"b{p}"] = dict(win=int((a > w + 0.05).sum()), loss=int((a < w - 0.05).sum()), deficit=float((w - a).max()))

# Vargha-Delaney A12 on scenario means
sm = pd.read_csv(os.path.join(M.RESULTS, "E1_scenario_means.csv"))
piv = sm.pivot(index="scenario", columns="method", values="HV")
def a12(x, y):
    x, y = np.asarray(x)[:, None], np.asarray(y)[None, :]
    return float(((x > y).sum() + 0.5 * (x == y).sum()) / (x.size * y.size))
st["a12"] = {m: a12(piv[M.PROP], piv[m]) for m in piv.columns if m != M.PROP}

# MOEA/D runs reaching each budget
summ = json.load(open(os.path.join(M.RESULTS, "summary_v7.json")))
st["budget_n"] = {r["pct"]: {m: r.get(m + "_n") for m in M.METHODS} for r in summ["budget"]}

# energy anchor: offloaded UDs, gain over all-local, enumerated sets (evaluation scenarios of E1, E3 base, E4)
cache = M.load_cache()
keys = [k for k in cache if dict(k)["seed"] == 2026 or 6001 <= dict(k)["seed"] <= 6029]
rows = []
for k in keys:
    d, v = dict(k), cache[k]
    sc = Scenario(**d)
    allloc = float(np.sum(sc.kappa * sc.f_min ** 2 * sc.C))
    rows.append(dict(M=d["M"], ext=len(d) > 2, off=len(v["info"]["offload_set"]), gain=100 * (1 - v["E"] / allloc),
                     sets=v["info"].get("subsets", np.nan), cand=len(v["info"]["candidates"])))
df = pd.DataFrame(rows)
st["energy_anchor"] = dict(max_off=int(df.off.max()), sets_max=int(df.sets.max()), cand_max=int(df.cand.max()),
                           gain={int(m): [float(g.gain.min()), float(g.gain.max())] for m, g in df[~df.ext].groupby("M")})

# base scenario
base = [o for o in out if o["kw"]["seed"] == 2026 and o["name"] == M.PROP][0]
st["evals_base"] = int(base["evals"])
json.dump(st, open(os.path.join(M.RESULTS, "paper_stats.json"), "w"), indent=1)
print(json.dumps(st, indent=1))

"""
Statistics of the equal-wall-clock comparison (experiments_v10_walltime.py, E8).
For every scenario of E4 (M = 20, 30, 50, first ten evaluation seeds) and of the fresh M = 50 set, the reference front
is the non-dominated union of the seven methods of Table V and of the longer runs; HV, IGD+ and HV_2T* are computed
as in analysis_rev.py, the five runs of a stochastic method are averaged per scenario, and ACE-SQP is compared with
every other variant by the one-sided Wilcoxon signed-rank test with Holm correction.
Output: results/summary_v10.json
"""
import json, os, pickle
import numpy as np
import pandas as pd
import methods as M
from model import Scenario
from scipy.stats import wilcoxon
from analysis import metric_fns, holm, HV_


def hv_low(F, ref, T, fac=2.0):
    """As in analysis_rev.py (which runs its whole analysis on import)."""
    r = ref[ref[:, 0] <= fac * T]
    lo = np.array([T, r[:, 1].min()]); hi = np.array([fac * T, r[:, 1].max()])
    G = F[F[:, 0] <= fac * T]
    return float(HV_((G - lo) / (hi - lo))) if len(G) else 0.0


def paired(piv, prop, others, better="greater"):
    """As in analysis_rev.py (v11: extensions of ACE-SQP are outside the Holm family; their entry is the unadjusted
    one-sided p-value that the extension is better than ACE-SQP)."""
    fam = [m for m in others if m not in M.EXTENSIONS]
    raw = lambda a, b, alt: 1.0 if np.allclose(piv[a] - piv[b], 0) else wilcoxon(piv[a], piv[b], alternative=alt).pvalue
    adj = dict(zip(fam, holm([raw(prop, m, better) for m in fam]))) if fam else {}
    rev = "less" if better == "greater" else "greater"
    return [adj[m] if m in adj else raw(prop, m, rev) for m in others]


R = M.RESULTS
load = lambda n: pickle.load(open(os.path.join(R, f"raw_{n}.pkl"), "rb"))
cal = json.load(open(os.path.join(R, "v10_calib.json")))
SEVEN = [M.PROP, "SR-NSGA-II+SQP", "WS-SQP", "NBI-SQP", "AEC-SQP", "BCD-eps", "SR-NSGA-II", "NSGA-II+S", "MOEA/D+S"]   # v12: + NBI-SQP, AEC-SQP
S10 = [2026] + list(range(6001, 6010))
FRESH = list(range(9001, 9011))

base = load("e4") + load("e4b") + load("fresh50")
wt = load("v10wt")
rec = []          # (set, seed, label, run, F)
for o in base:
    if o["name"] in SEVEN:
        st = "50f" if o["kw"]["seed"] in FRESH else str(o["kw"]["M"])
        rec.append((st, o["kw"]["seed"], o["name"], o["seed"], o["F"], o["kw"]))
# the 300-generation fronts of the long runs must coincide with the E4 runs (same seeds and settings)
check = []
for o in wt:
    st = "50f" if o["kw"]["seed"] in FRESH else str(o["kw"]["M"])
    if o["name"].startswith("BCD"):
        rec.append((st, o["kw"]["seed"], "BCD-eps (equal time)", 0, o["F"], o["kw"]))
        continue
    g300, geq, g2 = o["G"]
    orig = [b for b in base if b["name"] == o["name"] and b["kw"] == o["kw"] and b["seed"] == o["seed"]][0]["F"]
    f = o["fronts"][300]
    check.append(bool(f.shape == orig.shape and np.allclose(np.sort(f, 0), np.sort(orig, 0))))
    rec.append((st, o["kw"]["seed"], f"{o['name']} (equal time)", o["seed"], o["fronts"][geq], o["kw"]))
    rec.append((st, o["kw"]["seed"], f"{o['name']} (2x time)", o["seed"], o["fronts"][g2], o["kw"]))

rows = []
for st in ("20", "30", "50", "50f"):
    for s in (FRESH if st == "50f" else S10):
        sub = [r for r in rec if r[0] == st and r[1] == s]
        sc = Scenario(**sub[0][5])
        T = M.latency_optimal(sc)[0]
        ref = M.nondom(np.vstack([r[4] for r in sub]))
        ev, _ = metric_fns(ref)
        for r in sub:
            h, g = ev(r[4])
            rows.append(dict(set=st, scenario=s, method=r[2], run=r[3], HV=h, IGD=g, HVlow=hv_low(r[4], ref, T)))
df = pd.DataFrame(rows).groupby(["set", "scenario", "method"]).mean(numeric_only=True).reset_index()
order = [M.PROP, "SR-NSGA-II+SQP", "WS-SQP", "NBI-SQP", "AEC-SQP", "BCD-eps", "BCD-eps (equal time)",
         "SR-NSGA-II", "SR-NSGA-II (equal time)", "SR-NSGA-II (2x time)",
         "NSGA-II+S", "NSGA-II+S (equal time)", "NSGA-II+S (2x time)", "MOEA/D+S"]
out = dict(repro_300=dict(n=len(check), identical=int(sum(check))), plan=cal["plan"], table=[])
for st in ("20", "30", "50", "50f"):
    g = df[df.set == st]
    for c in ("HV", "HVlow", "IGD"):
        pv = g.pivot(index="scenario", columns="method", values=c)
        oth = [m for m in order if m != M.PROP and m in pv.columns]
        pp = paired(pv, M.PROP, oth, "less" if c == "IGD" else "greater")
        for m in order:
            if m not in pv.columns:
                continue
            better = (pv[M.PROP] < pv[m]) if c == "IGD" else (pv[M.PROP] > pv[m])
            out["table"].append(dict(set=st, metric=c, method=m, mean=float(pv[m].mean()), sd=float(pv[m].std()),
                                     wins=int(better.sum()) if m != M.PROP else None,
                                     p=float(pp[oth.index(m)]) if m != M.PROP else None))
bcd = [o for o in wt if o["name"].startswith("BCD")]
out["bcd_equal_time"] = {st: dict(evals=float(np.mean([o["evals"] for o in bcd if ("50f" if o["kw"]["seed"] in FRESH else str(o["kw"]["M"])) == st])))
                         for st in ("20", "30", "50", "50f")}
json.dump(out, open(os.path.join(R, "summary_v10.json"), "w"), indent=1)
t = pd.DataFrame(out["table"])
print("300-generation fronts identical to E4:", out["repro_300"])
for c in ("HV", "HVlow", "IGD"):
    print(c); print(t[t.metric == c].pivot(index="method", columns="set", values="mean").reindex(order).round(4).to_string())
print("wins of ACE-SQP (HV)"); print(t[t.metric == "HV"].pivot(index="method", columns="set", values="wins").reindex(order).to_string())
print("Holm p (HV)"); print(t[t.metric == "HV"].pivot(index="method", columns="set", values="p").reindex(order).to_string())
print(json.dumps(out["bcd_equal_time"]))

"""
Statistics of the revision experiments (experiments_rev.py) and additional indicators on E1.
Output: results/summary_rev.json
"""
import json, os, pickle
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import methods as M
from model import Scenario, objectives
from sqp_core import eps_levels
from analysis import metric_fns, holm, HV_, load

R = M.RESULTS
CFG = json.load(open(os.path.join(R, "config.json")))
SHAPE = CFG["ace"]["shape"]
out = {}


def hv_low(F, ref, T, fac=2.0):
    """Hypervolume restricted to the operating region [T*, fac T*] (fac = 2 in the paper): both fronts are cut at
    fac T* and normalized by the reference front inside the region."""
    r = ref[ref[:, 0] <= fac * T]
    lo = np.array([T, r[:, 1].min()]); hi = np.array([fac * T, r[:, 1].max()])
    G = F[F[:, 0] <= fac * T]
    return float(HV_((G - lo) / (hi - lo))) if len(G) else 0.0


def paired(piv, prop, others, better="greater"):
    """Holm-adjusted one-sided p-values of ACE-SQP against the independent methods. For an extension of ACE-SQP
    (M.EXTENSIONS) the entry is instead the unadjusted one-sided p-value that the extension is better than ACE-SQP."""
    fam = [m for m in others if m not in M.EXTENSIONS]
    raw = lambda a, b, alt: 1.0 if np.allclose(piv[a] - piv[b], 0) else wilcoxon(piv[a], piv[b], alternative=alt).pvalue
    adj = dict(zip(fam, holm([raw(prop, m, better) for m in fam]))) if fam else {}
    rev = "less" if better == "greater" else "greater"
    return [adj[m] if m in adj else raw(prop, m, rev) for m in others]


def min_e(F, x):
    ok = F[F[:, 0] <= x * (1 + 1e-6), 1]
    return ok.min() if len(ok) else np.nan


def trap_excess(F_full, F_noact, sc):
    """Energy excess (%) of the front without activation over the full front at the ladder levels."""
    T = M.latency_optimal(sc)[0]
    lev = eps_levels(T, M.t_nadir(sc), M.K_FRONT, SHAPE)
    ex = np.array([100 * (min_e(F_noact, x) / min_e(F_full, x) - 1) for x in lev])
    return ex


def idle_at_anchor(sc):
    T = M.latency_optimal(sc)[0]
    return int((sc.C / sc.f_maxm <= T).sum())


# ------------------------------------------------------------------ E1: restricted HV and trap statistics
M.init_worker()
e1 = load("e1")
scen = sorted({o["kw"]["seed"] for o in e1})
rows = []
for s in scen:
    sc = Scenario(M=10, seed=s)
    T = M.latency_optimal(sc)[0]
    sub = [o for o in e1 if o["kw"]["seed"] == s]
    ref = M.nondom(np.vstack([o["F"] for o in sub]))
    for o in sub:
        rows.append(dict(scenario=s, method=o["name"], HVlow=hv_low(o["F"], ref, T)))
df = pd.DataFrame(rows).groupby(["scenario", "method"]).mean().reset_index()
piv = df.pivot(index="scenario", columns="method", values="HVlow")
others = [m for m in M.METHODS if m != M.PROP]
p = paired(piv, M.PROP, others)
out["e1_hvlow"] = {m: dict(mean=float(piv[m].mean()), sd=float(piv[m].std()),
                           wins=int((piv[M.PROP] > piv[m]).sum()) if m != M.PROP else None,
                           p=float(p[others.index(m)]) if m != M.PROP else None) for m in M.METHODS}
# sensitivity of the restricted HV to the width of the region (1.5 T* and 3 T*)
out["e1_hvlow_sens"] = {}
for fac in (1.5, 3.0):
    rows = []
    for s in scen:
        sc = Scenario(M=10, seed=s)
        T = M.latency_optimal(sc)[0]
        sub = [o for o in e1 if o["kw"]["seed"] == s]
        ref = M.nondom(np.vstack([o["F"] for o in sub]))
        for o in sub:
            rows.append(dict(scenario=s, method=o["name"], v=hv_low(o["F"], ref, T, fac)))
    pv = pd.DataFrame(rows).groupby(["scenario", "method"]).mean().reset_index().pivot(index="scenario", columns="method", values="v")
    pp = paired(pv, M.PROP, others)
    out["e1_hvlow_sens"][str(fac)] = {m: dict(mean=float(pv[m].mean()), wins=int((pv[M.PROP] > pv[m]).sum()),
                                              p=float(pp[others.index(m)])) for m in others}
    out["e1_hvlow_sens"][str(fac)][M.PROP] = dict(mean=float(pv[M.PROP].mean()))

abl = load("abl")
ex_all, idle = [], []
for s in scen:
    sc = Scenario(M=10, seed=s)
    full = [o for o in e1 if o["name"] == M.PROP and o["kw"]["seed"] == s][0]["F"]
    noact = [a for a in abl if a["label"] == "no activation" and a["s"] == s][0]["F"]
    ex = trap_excess(full, noact, sc)
    ex_all.append(dict(s=s, max=float(np.nanmax(ex)), mean=float(np.nanmean(ex)), idle=idle_at_anchor(sc)))
tx = pd.DataFrame(ex_all)
out["e1_trap"] = dict(idle_mean=float(tx.idle.mean()), idle_max=int(tx.idle.max()), max_excess=float(tx["max"].max()),
                      n_over_1pct=int((tx["max"] > 1).sum()), n_over_5pct=int((tx["max"] > 5).sum()),
                      mean_of_max=float(tx["max"].mean()), n=len(tx))

# ------------------------------------------------------------------ STRESS
def block_stats(raw, names):
    """Per-scenario HV, IGD+, HV_2T*, paired tests against ACE-SQP, the energy saving within latency budgets and,
    when the variant without activation was run, the trap statistics."""
    rows, trap, bud = [], [], []
    for s in sorted({o["kw"]["seed"] for o in raw}):
        sub = [o for o in raw if o["kw"]["seed"] == s and o["name"] in names]
        sc = Scenario(**sub[0]["kw"])
        T, x_lat = M.latency_optimal(sc)
        E0 = float(objectives(x_lat[None], sc)[0, 1])
        ref = M.nondom(np.vstack([o["F"] for o in sub]))
        ev, _ = metric_fns(ref)
        for o in sub:
            h, g = ev(o["F"])
            rows.append(dict(scenario=s, method=o["name"], HV=h, IGD=g, HVlow=hv_low(o["F"], ref, T), evals=o["evals"] or M.TOTAL,
                             conv=o["conv"][0] / o["conv"][1] if o.get("conv") else np.nan, time=o["time"]))
            for pct in (10, 25):
                bud.append(dict(scenario=s, method=o["name"], pct=pct, save=100 * (1 - min_e(o["F"], T * (1 + pct / 100)) / E0)))
        fulls = [o for o in sub if o["name"] == M.PROP]
        noact = [o for o in sub if o["name"] == "ACE-SQP w/o activation"]
        if fulls and noact:
            ex = trap_excess(fulls[0]["F"], noact[0]["F"], sc)
            trap.append(dict(s=s, idle=idle_at_anchor(sc), max=float(np.nanmax(ex)), mean=float(np.nanmean(ex)), n_act=fulls[0]["n_act"]))
    sd = pd.DataFrame(rows).groupby(["scenario", "method"]).mean().reset_index()
    tab = []
    present = [m for m in names if m in set(sd.method)]
    for c in ("HV", "IGD", "HVlow"):
        pv = sd.pivot(index="scenario", columns="method", values=c)
        oth = [m for m in present if m != M.PROP]
        pp = paired(pv, M.PROP, oth, "less" if c == "IGD" else "greater")
        for m in present:
            tab.append(dict(metric=c, method=m, mean=float(pv[m].mean()), sd=float(pv[m].std()),
                            wins=int(((pv[M.PROP] < pv[m]) if c == "IGD" else (pv[M.PROP] > pv[m])).sum()) if m != M.PROP else None,
                            p=float(pp[oth.index(m)]) if m != M.PROP else None))
    ev_ = sd.groupby("method").agg(evals=("evals", "mean"), conv=("conv", "mean"), time=("time", "median")).reset_index().to_dict("records")
    bd = pd.DataFrame(bud).groupby(["scenario", "method", "pct"]).mean().reset_index().groupby(["method", "pct"]).save.mean()
    res = dict(table=tab, evals=ev_, budget={f"{m}|{p}": float(v) for (m, p), v in bd.items()}, n=int(sd.scenario.nunique()))
    if trap:
        tr = pd.DataFrame(trap)
        res.update(idle_mean=float(tr.idle.mean()), trap_max=float(tr["max"].max()), trap_mean_of_max=float(tr["max"].mean()),
                   trap_mean=float(tr["mean"].mean()), n_over_5pct=int((tr["max"] > 5).sum()), per_scenario=trap)
    return res


BLOCK_NAMES = [M.PROP, "ACE-SQP w/o activation", "BCD-eps", "WS-SQP", "NBI-SQP", "AEC-SQP", "SR-NSGA-II+SQP", "SR-NSGA-II"]
out["stress"] = block_stats(load("stress"), BLOCK_NAMES)
out["path"] = block_stats(load("path"), BLOCK_NAMES)
_snr = np.concatenate([10 * np.log10(Scenario(**o["kw"]).gamma) for o in load("path") if o["name"] == M.PROP])
out["path_snr"] = [float(_snr.min()), float(_snr.max())]
SEVEN = [M.PROP, "SR-NSGA-II+SQP", "WS-SQP", "NBI-SQP", "AEC-SQP", "BCD-eps", "SR-NSGA-II", "NSGA-II+S", "MOEA/D+S"]   # v12: + NBI-SQP, AEC-SQP
out["fresh50"] = block_stats(load("fresh50"), SEVEN)

# ------------------------------------------------------------------ E4 extended (M = 10, 20, 30, 50)
SIX = SEVEN          # the six strongest methods and BCD-eps
e4, e4b = load("e4"), load("e4b")
S10 = [2026] + list(range(6001, 6010))
allr = [o for o in e1 if o["kw"]["seed"] in S10] + e4 + e4b
rows = []
for m_ in (10, 20, 30, 50):
    for s in S10:
        sub = [o for o in allr if o["kw"]["M"] == m_ and o["kw"]["seed"] == s and o["name"] in SIX]
        if not sub:
            continue
        sc = Scenario(M=m_, seed=s)
        T = M.latency_optimal(sc)[0]
        ref = M.nondom(np.vstack([o["F"] for o in sub]))
        ev, _ = metric_fns(ref)
        for o in sub:
            h, g = ev(o["F"])
            rows.append(dict(M=m_, scenario=s, method=o["name"], HV=h, IGD=g, HVlow=hv_low(o["F"], ref, T), evals=o["evals"] or M.TOTAL))
d4 = pd.DataFrame(rows).groupby(["M", "scenario", "method"]).mean().reset_index()
tab4 = []
for m_ in (10, 20, 30, 50):
    g = d4[d4.M == m_]
    pv = g.pivot(index="scenario", columns="method", values="HV")
    meths = [x for x in SIX if x in pv.columns]
    oth = [x for x in meths if x != M.PROP]
    pp = paired(pv, M.PROP, oth)
    pl = g.pivot(index="scenario", columns="method", values="HVlow")
    for x in meths:
        q = g[g.method == x]
        tab4.append(dict(M=m_, method=x, HV=float(q.HV.mean()), HV_sd=float(q.HV.std()), IGD=float(q.IGD.mean()),
                         HVlow=float(q.HVlow.mean()), evals=float(q.evals.mean()), n=len(q),
                         wins=int((pv[M.PROP] > pv[x]).sum()) if x != M.PROP else None,
                         wins_low=int((pl[M.PROP] > pl[x]).sum()) if x != M.PROP else None,
                         p=float(pp[oth.index(x)]) if x != M.PROP else None))
conv50 = [o["conv"] for o in e4b if o["name"] == M.PROP and o["kw"]["M"] == 50]
# the M = 50 scenario with the fewest converged SLSQP solves, and the diagnostic run with a doubled per-level allowance
worst = [o for o in e4b if o["name"] == M.PROP and o["kw"]["M"] == 50 and o["kw"]["seed"] == 2026][0]   # failed in round 1
wrow = d4[(d4.M == 50) & (d4.scenario == worst["kw"]["seed"])].set_index("method").HV
diag_path = os.path.join(R, "diag_m50_2026.json")
diag = json.load(open(diag_path)) if os.path.exists(diag_path) else None
cache = M.load_cache()
a50 = [v for k, v in cache.items() if dict(k).get("M") == 50 and set(dict(k)) == {"M", "seed"}]
out["e4x"] = dict(table=tab4, conv50=[int(sum(c[0] for c in conv50)), int(sum(c[1] for c in conv50))],
                  worst50=dict(seed=int(worst["kw"]["seed"]), conv=list(worst["conv"]), evals=int(worst["evals"]),
                               HV={k: float(v) for k, v in wrow.items()}, diag=diag),
                  anchors50=dict(t_lat=float(np.mean([v["t_lat"] for v in a50])), t_en=float(np.mean([v["t_en"] for v in a50])),
                                 sets_max=int(max(v["info"]["subsets"] for v in a50)), cand_max=int(max(len(v["info"]["candidates"]) for v in a50))),
                  time_par={f"{m_}|{x}": float(np.median([o["time"] for o in allr if o["kw"]["M"] == m_ and o["name"] == x]))
                            for m_ in (20, 30, 50) for x in SIX if any(o["kw"]["M"] == m_ and o["name"] == x for o in allr)})

# ------------------------------------------------------------------ ablation with the finite-difference variant of larger budget
fd = load("fdbig")
rows = []
for s in scen:
    g = [a for a in abl if a["s"] == s]
    f_ = [o for o in fd if o["kw"]["seed"] == s][0]
    ace = [o for o in e1 if o["name"] == M.PROP and o["kw"]["seed"] == s][0]
    ref = M.nondom(np.vstack([a["F"] for a in g] + [ace["F"], f_["F"]]))
    ev, _ = metric_fns(ref)
    h, i_ = ev(ace["F"])
    rows.append(dict(s=s, label=M.PROP, HV=h, IGD=i_, evals=ace["evals"], conv=sum(x["status"] == 0 for x in ace["stats"]["levels"]),
                     levels=len(ace["stats"]["levels"]), time=ace["time"]))
    for a in g:
        h, i_ = ev(a["F"])
        rows.append(dict(s=s, label=a["label"], HV=h, IGD=i_, evals=a["evals"], conv=a["conv"], levels=a["levels"], time=a["time"]))
    h, i_ = ev(f_["F"])
    rows.append(dict(s=s, label="finite differences, budget x(4M+1)", HV=h, IGD=i_, evals=f_["evals"], conv=f_["conv"][0],
                     levels=f_["conv"][1], time=f_["time"]))
a = pd.DataFrame(rows)
out["ablation"] = a.groupby("label").agg(HV=("HV", "mean"), IGD=("IGD", "mean"), evals=("evals", "mean"), conv=("conv", "sum"),
                                         levels=("levels", "sum"), time=("time", "median")).reset_index().to_dict("records")

# ------------------------------------------------------------------ exact (SCIP)
if not os.path.exists(os.path.join(R, "raw_exact.pkl")):          # draft run before EXACT has finished
    json.dump(out, open(os.path.join(R, "summary_rev.json"), "w"), indent=1, default=float)
    raise SystemExit("raw_exact.pkl missing: summary written without the exact block")
ex = pd.DataFrame(load("exact"))
ex["excess"] = 100 * (ex.ace / ex.primal - 1)            # ACE-SQP above the best SCIP solution
ex["cert"] = 100 * (ex.ace - ex.dual) / ex.ace           # certified gap of ACE-SQP
ex["solved"] = ex.status.isin(["optimal", "gaplimit"])
out["exact"] = {int(m_): dict(cases=len(g), solved=int(g.solved.sum()), excess_max=float(g.excess.max()),
                              excess_mean=float(g.excess.mean()), better_than_ace=int((g.excess > 0.1).sum()),
                              cert_mean=float(g.cert.mean()), cert_max=float(g.cert.max()),
                              cert_solved_max=float(g[g.solved].cert.max()) if g.solved.any() else None,
                              time_median=float(g.time.median()))
                for m_, g in ex.groupby("M")}
out["exact_all"] = dict(cases=len(ex), solved=int(ex.solved.sum()), excess_max=float(ex.excess.max()),
                        better_than_ace=int((ex.excess > 0.1).sum()))

# ------------------------------------------------------------------ figure: the activation trap on a heterogeneous scenario
import matplotlib.pyplot as plt
from analysis import STYLE, COLW, save
st_raw = load("stress")
worst_s = max(out["stress"]["per_scenario"], key=lambda r: r["mean"])["s"]
sub = [o for o in st_raw if o["kw"]["seed"] == worst_s]
sc = Scenario(**sub[0]["kw"])
T = M.latency_optimal(sc)[0]
fig, axs = plt.subplots(1, 2, figsize=(COLW * 2.03, 2.25), gridspec_kw=dict(width_ratios=[1.2, 1]))
sty = {M.PROP: (STYLE[M.PROP][0], "-", "ACE-SQP"), "ACE-SQP w/o activation": ("#eb6834", "--", "ACE-SQP w/o activation"),
       "BCD-eps": (STYLE["BCD-eps"][0], ":", r"BCD-$\varepsilon$")}
for name, (col, ls, lab) in sty.items():
    F = [o for o in sub if o["name"] == name][0]["F"]
    F = F[F[:, 0] <= 3 * T]
    axs[0].plot(F[:, 0] / T, F[:, 1], ls=ls, color=col, lw=1.4, label=lab)
axs[0].set_xlabel(r"Latency $J_1/T^\star$"); axs[0].set_ylabel(r"Total device energy $J_2$ (J)")
axs[0].set_title("(a) Fronts up to $3T^\\star$", fontsize=8, loc="left"); axs[0].legend(fontsize=6.3, frameon=False)
lev = eps_levels(T, M.t_nadir(sc), M.K_FRONT, SHAPE)
full = [o for o in sub if o["name"] == M.PROP][0]["F"]
for name, (col, ls, lab) in list(sty.items())[1:]:
    F = [o for o in sub if o["name"] == name][0]["F"]
    ex = [100 * (min_e(F, x) / min_e(full, x) - 1) for x in lev]
    axs[1].plot(lev / T, ex, ls=ls, color=col, lw=1.4, label=lab)
axs[1].set_xscale("log"); axs[1].set_xlabel(r"Latency level $\varepsilon/T^\star$"); axs[1].set_ylabel("Energy above ACE-SQP (%)")
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
_tk = [t for t in (1, 1.5, 2, 3, 5, 8) if t <= lev[-1] / T]
axs[1].xaxis.set_major_locator(FixedLocator(_tk)); axs[1].xaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in _tk]))
axs[1].xaxis.set_minor_locator(NullLocator())
axs[1].set_title("(b) Excess energy per level", fontsize=8, loc="left")
fig.tight_layout(w_pad=1.0)
save(fig, "fig_trap")
out["fig_trap_seed"] = int(worst_s)

try:
    from pyscipopt import Model as _SM
    out["scip_version"] = str(_SM().version())
except Exception:
    out["scip_version"] = "?"
json.dump(out, open(os.path.join(R, "summary_rev.json"), "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float)[:15000])

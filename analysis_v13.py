"""
V13 analysis: diagnostic variants (experiments_v13.py) and the reason for the losses of NBI-SQP on E1.
For every set, the reference front of a scenario is the union of the compared group (ACE-SQP, ACE-SQP w/o activation
where available, NBI-SQP, AEC-SQP, WS-SQP and the three variants), so HV, HV_2T* and IGD+ differ slightly from the
main tables (the normalization uses the extreme points of this reference front). Stochastic variants: mean of five
runs. Run times in the raw files come from a parallel pool; sequential times: timing_v13.py.
Output: results/summary_v13.json
"""
import json, os, pickle
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, spearmanr
import methods as M
from model import Scenario, objectives
from analysis import metric_fns, holm, load

R = M.RESULTS
BASE = [M.PROP, "ACE-SQP w/o activation", "NBI-SQP", "AEC-SQP", "WS-SQP"]
GROUP = BASE + M.VARIANTS


def hv_low(F, ref, T, fac=2.0):
    from analysis import HV_
    r = ref[ref[:, 0] <= fac * T]
    lo = np.array([T, r[:, 1].min()]); hi = np.array([fac * T, r[:, 1].max()])
    G = F[F[:, 0] <= fac * T]
    return float(HV_((G - lo) / (hi - lo))) if len(G) else 0.0


def min_e(F, x):
    ok = F[F[:, 0] <= x * (1 + 1e-6), 1]
    return ok.min() if len(ok) else np.nan


def p_one(a, b, alt="greater"):
    d = np.asarray(a) - np.asarray(b)
    return 1.0 if np.allclose(d, 0) else float(wilcoxon(a, b, alternative=alt).pvalue)


def records(block):
    base = [o for o in load(block) if o["name"] in BASE]
    if block == "e1":                      # the variant without activation of E1 is in the ablation file
        for a in load("abl"):
            if a["label"] == "no activation":
                base.append(dict(name="ACE-SQP w/o activation", kw=dict(M=10, seed=a["s"]), seed=0, F=a["F"],
                                 evals=a.get("evals"), time=a.get("time", np.nan)))
    return base + pickle.load(open(os.path.join(R, f"raw_v13_{block}.pkl"), "rb"))


def block_stats(recs):
    rows = []
    groups = {}
    for o in recs:
        groups.setdefault(tuple(sorted(o["kw"].items())), []).append(o)
    for key, sub in groups.items():
        sc = Scenario(**dict(key))
        T, x_lat = M.latency_optimal(sc)
        E0 = float(objectives(x_lat[None], sc)[0, 1])
        ref = M.nondom(np.vstack([o["F"] for o in sub]))
        ev, _ = metric_fns(ref)
        for o in sub:
            h, g = ev(o["F"])
            rows.append(dict(scenario=str(key), M=sc.M, method=o["name"], HV=h, IGD=g, HVlow=hv_low(o["F"], ref, T),
                             s10=100 * (1 - min_e(o["F"], 1.10 * T) / E0), s25=100 * (1 - min_e(o["F"], 1.25 * T) / E0),
                             evals=o.get("evals") or M.TOTAL, time=o.get("time", np.nan)))
    d = pd.DataFrame(rows).groupby(["scenario", "M", "method"]).mean().reset_index()
    out = {}
    for Mv, dm in d.groupby("M"):
        present = [m for m in GROUP if m in set(dm.method)]
        if M.PROP not in present or not set(M.VARIANTS) & set(present):
            continue                       # extra M of a main file without ACE-SQP or without the variants
        res = dict(n=int(dm.scenario.nunique()), means={}, vs_ace={}, pairs={})
        for c in ("HV", "HVlow", "IGD", "s10", "s25", "evals", "time"):
            pv = dm.pivot(index="scenario", columns="method", values=c)
            res["means"][c] = {m: float(pv[m].mean()) for m in present}
        for c in ("HV", "HVlow"):
            pv = dm.pivot(index="scenario", columns="method", values=c)
            oth = [m for m in present if m != M.PROP]
            pp = holm([p_one(pv[M.PROP], pv[m]) for m in oth])
            res["vs_ace"][c] = {m: dict(ace_better=int((pv[M.PROP] > pv[m] + 1e-12).sum()), p_holm=float(p))
                                for m, p in zip(oth, pp)}
            # questions of the reviewer: does the activation restart help NBI-SQP; is it better than a random restart
            pr = {}
            if "NBI-SQP+act" in pv:
                pr["NBI+act better than NBI"] = dict(n=int((pv["NBI-SQP+act"] > pv["NBI-SQP"] + 1e-12).sum()),
                                                     p=p_one(pv["NBI-SQP+act"], pv["NBI-SQP"]),
                                                     gain=float((pv["NBI-SQP+act"] - pv["NBI-SQP"]).mean()))
                # strict count (no ties) used in Section VI-I of the paper ("in all 30 E1 scenarios", "all ten with M = 20")
                pr["NBI+act strictly higher than ACE"] = dict(n=int((pv["NBI-SQP+act"] > pv[M.PROP] + 1e-12).sum()),
                                                             ties=int(((pv["NBI-SQP+act"] - pv[M.PROP]).abs() <= 1e-12).sum()))
            if "ACE-SQP (random restart)" in pv:
                pr["activation better than random restart"] = dict(
                    n=int((pv[M.PROP] > pv["ACE-SQP (random restart)"] + 1e-12).sum()),
                    n_worse=int((pv[M.PROP] < pv["ACE-SQP (random restart)"] - 1e-12).sum()),
                    p=p_one(pv[M.PROP], pv["ACE-SQP (random restart)"]),
                    p_rev=p_one(pv["ACE-SQP (random restart)"], pv[M.PROP]),
                    diff=float((pv[M.PROP] - pv["ACE-SQP (random restart)"]).mean()))
            if "ACE-SQP w/o activation" in pv and "ACE-SQP (random restart)" in pv:
                pr["random restart better than none"] = dict(
                    n=int((pv["ACE-SQP (random restart)"] > pv["ACE-SQP w/o activation"] + 1e-12).sum()),
                    gain=float((pv["ACE-SQP (random restart)"] - pv["ACE-SQP w/o activation"]).mean()))
            res["pairs"][c] = pr
        out[str(Mv)] = res
    return out


def e1_diagnosis():
    """Why does NBI-SQP lose in some E1 scenarios? Compare its HV deficit with that of ACE-SQP without the restart and
    with the number of UDs idle at the latency anchor."""
    e1 = [o for o in load("e1") if o["name"] in (M.PROP, "NBI-SQP")]
    abl = {a["s"]: a["F"] for a in load("abl") if a["label"] == "no activation"}
    rows = []
    for s in sorted({o["kw"]["seed"] for o in e1}):
        sc = Scenario(M=10, seed=s)
        F = {o["name"]: o["F"] for o in e1 if o["kw"]["seed"] == s}
        ref = M.nondom(np.vstack(list(F.values()) + [abl[s]]))
        ev, _ = metric_fns(ref)
        T = M.latency_optimal(sc)[0]
        rows.append(dict(s=s, d_nbi=ev(F[M.PROP])[0] - ev(F["NBI-SQP"])[0], d_noact=ev(F[M.PROP])[0] - ev(abl[s])[0],
                         idle=int((sc.C / sc.f_maxm <= T).sum())))
    d = pd.DataFrame(rows)
    top = d.sort_values("d_nbi", ascending=False).head(5)
    rho = spearmanr(d.d_nbi, d.d_noact)
    return dict(diff_median=float(d.d_nbi.median()), diff_mean=float(d.d_nbi.mean()), diff_max=float(d.d_nbi.max()),
                nbi_better=[int(x) for x in d[d.d_nbi < 0].s], top5=top.to_dict("records"),
                spearman_nbi_vs_noact=dict(rho=float(rho.correlation), p=float(rho.pvalue)),
                share_of_total_from_top5=float(top.d_nbi.clip(lower=0).sum() / d.d_nbi.clip(lower=0).sum()))


def e1_hvlow_vs_nbi():
    """Distribution of the HV_2T* differences between ACE-SQP and NBI-SQP on E1, with the reference front of Table 4
    (union of all methods and runs of the scenario), as stated in Section VI-B of the paper."""
    e1 = load("e1")
    rows = []
    for s in sorted({o["kw"]["seed"] for o in e1}):
        sub = [o for o in e1 if o["kw"]["seed"] == s]
        ref = M.nondom(np.vstack([o["F"] for o in sub]))
        T = M.latency_optimal(Scenario(M=10, seed=s))[0]
        v = {n: hv_low([o["F"] for o in sub if o["name"] == n][0], ref, T) for n in (M.PROP, "NBI-SQP")}
        rows.append(dict(s=s, d=v["NBI-SQP"] - v[M.PROP], ace=v[M.PROP], nbi=v["NBI-SQP"]))
    d = pd.DataFrame(rows)
    hi, lo = d[d.d > 0], d[d.d < 0]
    return dict(mean_ace=float(d.ace.mean()), mean_nbi=float(d.nbi.mean()), n_nbi_higher=int(len(hi)),
                nbi_higher_median=float(hi.d.median()), nbi_higher_max=float(hi.d.max()),
                nbi_lower_min=float(-lo.d.max()), nbi_lower_max=float(-lo.d.min()))


if __name__ == "__main__":
    M.init_worker()
    out = {"e1_diagnosis": e1_diagnosis(), "e1_hvlow_vs_nbi": e1_hvlow_vs_nbi()}
    for b in ("e1", "stress", "path", "e4", "e4b", "fresh50"):
        if os.path.exists(os.path.join(R, f"raw_v13_{b}.pkl")):
            out[b] = block_stats(records(b))
    json.dump(out, open(os.path.join(R, "summary_v13.json"), "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float)[:20000])

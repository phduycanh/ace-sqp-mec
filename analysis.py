"""
Tables, statistics and figures of the paper from results/raw_*.pkl.
Scenario-level statistics: each scenario is one paired observation (mean over its runs).
Output: results/summary_v7.json, results/E1_scenario_means.csv, results/fig_*.pdf|png
"""
import json, os, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, friedmanchisquare
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus
import methods as M
from model import Scenario, objectives, components

R = M.RESULTS
load = lambda n: pickle.load(open(os.path.join(R, f"raw_{n}.pkl"), "rb"))
HV_ = HV(ref_point=np.array([1.1, 1.1]))
COLW, TEXTW = 3.45, 7.0
plt.rcParams.update({"font.size": 8, "font.family": "serif", "mathtext.fontset": "dejavuserif",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e4e3dd", "grid.linewidth": 0.6, "axes.edgecolor": "#8a8a85",
                     "lines.linewidth": 1.4, "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
STYLE = {"ACE-SQP": ("#2a78d6", "o"), "SR-NSGA-II+SQP": ("#7b5cc4", "h"), "SR-NSGA-II": ("#eb6834", "s"),
         "NSGA-II+S": ("#1baf7a", "^"), "NSGA-II": ("#8a8a85", "v"), "SPEA2+S": ("#eda100", "D"),
         "MOEA/D+S": ("#e87ba4", "<"), "WS-GA+S": ("#008300", "P"), "WS-SQP": ("#8c5a3c", "*"), "Random": ("#b5b5b0", "x"),
         "BCD-eps": ("#0e8fa3", "p")}


def save(fig, name):
    fig.savefig(os.path.join(R, name + ".pdf")); fig.savefig(os.path.join(R, name + ".png")); plt.close(fig)


def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); a = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * p[i])); a[i] = run
    return a


def metric_fns(ref):
    lo, hi = ref.min(0), ref.max(0)
    n = lambda F: (F - lo) / (hi - lo)
    igd = IGDPlus(n(ref))
    return (lambda F: (HV_(n(F)), igd(n(F)))), n


def e1_tables(out):
    info, rows, loo = {}, [], []
    scen = sorted({o["kw"]["seed"] for o in out})
    methods = M.METHODS
    for s in scen:
        sub = [o for o in out if o["kw"]["seed"] == s]
        ref = M.nondom(np.vstack([o["F"] for o in sub]))
        ev, _ = metric_fns(ref)
        # leave-one-out front: neither the proposed method nor its extensions contribute to the reference
        ref_no = M.nondom(np.vstack([o["F"] for o in sub if o["name"] != M.PROP and o["name"] not in M.EXTENSIONS]))
        ev_no, _ = metric_fns(ref_no)
        for o in sub:
            h, g = ev(o["F"])
            h2, g2 = ev_no(o["F"])
            rows.append(dict(scenario=s, method=o["name"], seed=o["seed"], HV=h, IGD=g, HV_loo=h2, IGD_loo=g2,
                             minT=o["F"][:, 0].min(), minE=o["F"][:, 1].min(), pts=len(o["F"]), evals=o["evals"] or M.TOTAL))
    df = pd.DataFrame(rows)
    sm = df.groupby(["scenario", "method"]).mean(numeric_only=True).reset_index()
    sm.to_csv(os.path.join(R, "E1_scenario_means.csv"), index=False)
    piv = {c: sm.pivot(index="scenario", columns="method", values=c)[methods] for c in ("HV", "IGD", "HV_loo", "IGD_loo")}
    indep = [m for m in methods if m not in M.EXTENSIONS]       # independent methods: test family and Friedman ranks
    others = [m for m in indep if m != M.PROP]
    p_hv = holm([wilcoxon(piv["HV"][M.PROP], piv["HV"][m], alternative="greater").pvalue for m in others])
    p_igd = holm([wilcoxon(piv["IGD"][M.PROP], piv["IGD"][m], alternative="less").pvalue for m in others])
    p_igd_loo = holm([wilcoxon(piv["IGD_loo"][M.PROP], piv["IGD_loo"][m], alternative="less").pvalue for m in others])
    fried = friedmanchisquare(*[piv["HV"][m] for m in indep])
    ranks = (-piv["HV"][indep]).rank(axis=1).mean()
    tab = []
    for m in methods:
        d = sm[sm.method == m]
        row = dict(method=m, HV=d.HV.mean(), HV_sd=d.HV.std(), IGD=d.IGD.mean(), IGD_sd=d.IGD.std(), IGD_loo=d.IGD_loo.mean(),
                   rank=float(ranks[m]) if m in indep else None, pts=d.pts.mean(), evals=d.evals.mean(),
                   gapT=float(((d.set_index("scenario").minT) / pd.Series({s: M.latency_optimal(Scenario(M=10, seed=s))[0] for s in scen}) - 1).mean() * 100),
                   gapE=float(((d.set_index("scenario").minE) / pd.Series({s: M.energy_optimal(Scenario(M=10, seed=s))[0] for s in scen}) - 1).mean() * 100),
                   wins_hv=int((piv["HV"][M.PROP] > piv["HV"][m]).sum()) if m != M.PROP else None)
        if m in others:
            i = others.index(m)
            row.update(p_hv=float(p_hv[i]), p_igd=float(p_igd[i]), p_igd_loo=float(p_igd_loo[i]))
        elif m in M.EXTENSIONS:          # does the extension improve on ACE-SQP? (one-sided, unadjusted)
            dlt = piv["HV"][m] - piv["HV"][M.PROP]
            row.update(ext_better=int((dlt > 0).sum()), ext_equal=int((dlt == 0).sum()), ext_gain_mean=float(dlt.mean()),
                       ext_gain_max=float(dlt.max()), p_ext=float(wilcoxon(piv["HV"][m], piv["HV"][M.PROP], alternative="greater").pvalue))
        tab.append(row)
    info["table"] = tab
    info["friedman"] = dict(stat=float(fried.statistic), p=float(fried.pvalue), n=len(scen))
    # within-scenario spread of stochastic methods (mean over scenarios of the SD over runs)
    info["within_sd"] = {m: float(df[df.method == m].groupby("scenario").HV.std().mean()) for m in others}
    # SQP convergence of ACE-SQP
    st = [o["stats"] for o in out if o["name"] == M.PROP]
    info["ace"] = dict(converged=int(sum(x["status"] == 0 for s_ in st for x in s_["levels"])), solves=int(sum(len(s_["levels"]) for s_ in st)),
                       kept=float(np.mean([s_["n_kept"] for s_ in st])), activated=float(np.mean([s_["n_activated"] for s_ in st])),
                       evals_mean=float(np.mean([o["evals"] for o in out if o["name"] == M.PROP])),
                       evals_max=int(max(o["evals"] for o in out if o["name"] == M.PROP)))
    # latency budgets relative to T*, per scenario
    budget = []
    for pct in (5, 10, 25, 50, 100):
        row = dict(pct=pct)
        for m in methods:
            vals = []
            for s in scen:
                sc = Scenario(M=10, seed=s)
                T, x = M.latency_optimal(sc)
                E0 = objectives(x[None], sc)[0, 1]
                per = []
                for o in out:
                    if o["name"] == m and o["kw"]["seed"] == s:
                        ok = o["F"][o["F"][:, 0] <= T * (1 + pct / 100)]
                        if len(ok):
                            per.append(100 * (1 - ok[:, 1].min() / E0))
                if per:
                    vals.append(np.mean(per))
            row[m] = float(np.mean(vals)) if vals else None
            row[m + "_n"] = len(vals)
        budget.append(row)
    info["budget"] = budget
    return info, df, sm


def base_figs(out, info):
    sub = [o for o in out if o["kw"]["seed"] == 2026]
    ref = M.nondom(np.vstack([o["F"] for o in sub]))
    ev, _ = metric_fns(ref)
    T_star = M.latency_optimal(Scenario(M=10, seed=2026))[0]
    fig, axs = plt.subplots(1, 2, figsize=(TEXTW, 2.9), gridspec_kw=dict(width_ratios=[1.15, 1]))
    shown = ["ACE-SQP", "SR-NSGA-II+SQP", "WS-SQP", "BCD-eps", "SR-NSGA-II", "NSGA-II"]   # the others are in Table E1
    names = {"SR-NSGA-II+SQP": "ACE-SQP+EA", "BCD-eps": r"BCD-$\varepsilon$"}
    for ax, zoom in zip(axs, (False, True)):
        for m in shown:
            if zoom and m == "Random":
                continue
            runs = [o for o in sub if o["name"] == m]
            hv = [ev(o["F"])[0] for o in runs]
            F = runs[int(np.argsort(hv)[len(hv) // 2])]["F"]
            c, mk = STYLE[m]
            dense = len(F) > 15
            ax.plot(F[:, 0], F[:, 1], ls="-" if dense else "none", lw=1.1, marker=mk, ms=2.5 if dense else 5,
                    markevery=max(1, len(F) // 14), color=c, label=names.get(m, m), mec="white" if mk not in "xP*" else c, mew=0.4)
        ax.axvline(T_star, color="#55554f", lw=0.8, ls="--")
        ax.set_xlabel(r"Minimax latency $J_1$ (s)")
    axs[0].set_xlim(0, 65); axs[0].set_ylim(0, None); axs[0].set_ylabel(r"Total device energy $J_2$ (J)")
    axs[0].set_title("(a) Whole front", fontsize=8, loc="left")
    axs[1].set_xlim(T_star * 0.95, 22); axs[1].set_ylim(10, 90)
    axs[1].set_title("(b) Low-latency region", fontsize=8, loc="left")
    axs[0].legend(fontsize=6.0, frameon=False, loc="upper right")
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_pareto")
    # convergence on the base scenario; ACE-SQP is drawn at its final evaluation count
    fig, ax = plt.subplots(figsize=(COLW, 2.35))
    for m in ("SR-NSGA-II+SQP", "SR-NSGA-II", "NSGA-II+S", "NSGA-II", "SPEA2+S", "MOEA/D+S"):
        curves = [[(n, ev(M.nondom(F))[0]) for n, F in o["hist"]] for o in sub if o["name"] == m]
        L = min(len(c) for c in curves)
        arr = np.array([[v for _, v in c[:L]] for c in curves])
        x = np.array([n for n, _ in curves[0][:L]])
        c, mk = STYLE[m]
        ax.plot(x, arr.mean(0), color=c, marker=mk, markevery=max(1, L // 7), ms=3, mec="white", mew=0.4,
                label={"SR-NSGA-II+SQP": "ACE-SQP+EA"}.get(m, m))
    a = [o for o in sub if o["name"] == M.PROP][0]
    ax.axhline(ev(a["F"])[0], color=STYLE[M.PROP][0], lw=1.0, ls="--")
    ax.plot([a["evals"]], [ev(a["F"])[0]], "o", color=STYLE[M.PROP][0], ms=5, label=f"{M.PROP} (finished)", zorder=5)
    ax.set_xlabel("Function evaluations"); ax.set_ylabel("Normalized hypervolume")
    ax.legend(fontsize=5.8, frameon=False, loc="lower right")
    save(fig, "fig_convergence")
    # trade-off on the ACE-SQP front of the base scenario
    sc = Scenario(M=10, seed=2026)
    T, x = M.latency_optimal(sc)
    E0 = objectives(x[None], sc)[0, 1]
    F = a["F"]; X = a["X"]
    trade = []
    for pct in (5, 10, 25, 50, 100):
        ok = F[F[:, 0] <= T * (1 + pct / 100)]
        p = ok[np.argmin(ok[:, 1])]
        trade.append(dict(point=f"+{pct}%", T=float(p[0]), E=float(p[1]), dT=100 * (p[0] / T - 1), save=100 * (1 - p[1] / E0)))
    n = (F - F.min(0)) / (F.max(0) - F.min(0))
    k = F[np.argmin(np.linalg.norm(n, axis=1))]
    trade.append(dict(point="knee", T=float(k[0]), E=float(k[1]), dT=100 * (k[0] / T - 1), save=100 * (1 - k[1] / E0)))
    trade.append(dict(point="E*", T=float(F[-1, 0]), E=float(F[-1, 1]), dT=100 * (F[-1, 0] / T - 1), save=100 * (1 - F[-1, 1] / E0)))
    info["base_tradeoff"] = dict(T_star=float(T), E_at_T=float(E0), points=trade)
    # per-UD decisions
    order = np.argsort(objectives(X, sc)[:, 0])
    Fx, Xs = objectives(X, sc)[order], X[order]
    picks = {"Latency optimum": 0,
             "+25% latency budget": int(np.where(Fx[:, 0] <= 1.25 * T)[0][np.argmin(Fx[Fx[:, 0] <= 1.25 * T, 1])]),
             "Knee point": int(np.argmin(np.linalg.norm((Fx - Fx.min(0)) / (Fx.max(0) - Fx.min(0)), axis=1)))}
    fig, axs = plt.subplots(2, 1, figsize=(COLW, 3.1), sharex=True)
    w, idx = 0.27, np.arange(sc.M)
    ops = {}
    for j, (lab, i) in enumerate(picks.items()):
        c = components(Xs[i][None], sc)
        ops[lab] = dict(T=float(Fx[i, 0]), E=float(Fx[i, 1]), P=c["P"][0].tolist(), f_rel=(c["f"][0] / sc.f_maxm).tolist(),
                        B=(c["B"][0] / 1e6).tolist(), F=(c["F"][0] / 1e9).tolist(), Tm=c["T"][0].tolist())
        col = ["#2a78d6", "#eb6834", "#1baf7a"][j]
        axs[0].bar(idx + (j - 1) * w, c["P"][0], w, color=col, label=f"{lab} ({Fx[i,0]:.2f} s, {Fx[i,1]:.1f} J)")
        axs[1].bar(idx + (j - 1) * w, c["f"][0] / sc.f_maxm, w, color=col)
    axs[0].set_ylabel("Offloaded fraction $P_m$"); axs[1].set_ylabel(r"$f_m / f_m^{\max}$")
    axs[1].set_xticks(idx, [str(i + 1) for i in idx]); axs[1].set_xlabel("UD index $m$")
    axs[0].legend(fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.4), ncol=1)
    for ax in axs:
        ax.set_ylim(0, 1.05); ax.grid(axis="x", visible=False)
    fig.tight_layout(h_pad=0.4)
    save(fig, "fig_decisions")
    info["decisions"] = ops
    info["scenario_table"] = [dict(m=i + 1, S=sc.TS[i] / 1e6, fmax=sc.f_maxm[i] / 1e9, gamma=10 * np.log10(sc.gamma[i])) for i in range(sc.M)]
    # the same 25% budget for SR-NSGA-II (median run) on the base scenario
    srs = [o for o in sub if o["name"] == "SR-NSGA-II"]
    info["sr_at_25"] = float(np.mean([o["F"][o["F"][:, 0] <= 1.25 * T, 1].min() for o in srs]))


def e3(out):
    rows = []
    for o in out:
        kw = o["kw"]
        sc = Scenario(**kw)
        T = M.latency_optimal(sc)[0]
        E = M.energy_optimal(sc)
        x = M.latency_optimal(sc)[1]
        rows.append(dict(seed=kw["seed"], F=kw.get("F_max", 10e9) / 1e9, B=kw.get("B_max", 20e6) / 1e6, T=T, E_star=E[0], E_at_T=float(objectives(x[None], sc)[0, 1]),
                         E_at_8=float(o["F"][o["F"][:, 0] <= 8.5, 1].min()) if (o["F"][:, 0] <= 8.5).any() else None,
                         n_off=len(E[2]["offload_set"])))
    df = pd.DataFrame(rows)
    g = df.groupby(["F", "B"]).agg(T=("T", "mean"), T_sd=("T", "std"), E_at_T=("E_at_T", "mean"), E_star=("E_star", "mean"), n_off=("n_off", "mean")).reset_index()
    # figure: base scenario fronts
    fig, axs = plt.subplots(1, 2, figsize=(TEXTW, 2.4), sharey=True)
    for ax, key, vals, lab, unit in ((axs[0], "F_max", (5.0, 10.0, 15.0), r"$F_{\max}$", "GHz"), (axs[1], "B_max", (10.0, 20.0, 40.0), r"$B_{\max}$", "MHz")):
        for v, col, mk in zip(vals, ("#eb6834", "#2a78d6", "#1baf7a"), "s^o"):
            mult = 1e9 if key == "F_max" else 1e6
            for o in out:
                kw = o["kw"]
                if kw["seed"] == 2026 and (kw.get(key) == v * mult or (v in (10.0, 20.0) and v * mult == getattr(Scenario, key) and set(kw) == {"M", "seed"})):
                    T = M.latency_optimal(Scenario(**kw))[0]
                    ax.plot(o["F"][:, 0], o["F"][:, 1], color=col, marker=mk, markevery=max(1, len(o["F"]) // 10), ms=3.2, mec="white", mew=0.4,
                            label=f"{lab} = {v:g} {unit} ($T^\\star$ = {T:.2f} s)")
                    break
        ax.set_xlim(5, 30); ax.set_xlabel("Minimax latency (s)"); ax.legend(fontsize=6.6, frameon=False)
        ax.set_title(f"({'a' if key == 'F_max' else 'b'}) Effect of {lab}", fontsize=8, loc="left")
    axs[0].set_ylabel("Total device energy (J)")
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_sensitivity")
    return g.to_dict("records")


def e4(out, e1_out):
    rows = []
    for m_ in (10, 20, 30):
        src = e1_out if m_ == 10 else out
        scen = sorted({o["kw"]["seed"] for o in out})
        for s in scen:
            sub = [o for o in src if o["kw"]["M"] == m_ and o["kw"]["seed"] == s and o["name"] in (M.PROP, "SR-NSGA-II+SQP", "SR-NSGA-II", "NSGA-II+S")]
            ref = M.nondom(np.vstack([o["F"] for o in sub]))
            ev, _ = metric_fns(ref)
            for o in sub:
                h, g = ev(o["F"])
                rows.append(dict(M=m_, scenario=s, method=o["name"], HV=h, IGD=g, evals=o["evals"] or M.TOTAL))
    df = pd.DataFrame(rows)
    sm = df.groupby(["M", "scenario", "method"]).mean(numeric_only=True).reset_index()
    tab = sm.groupby(["M", "method"]).agg(HV=("HV", "mean"), HV_sd=("HV", "std"), IGD=("IGD", "mean"), evals=("evals", "mean")).reset_index().to_dict("records")
    st = [o for o in out if o["name"] == M.PROP]
    conv = {}
    for m_ in (20, 30):
        ss = [o["stats"] for o in st if o["kw"]["M"] == m_]
        conv[m_] = dict(converged=int(sum(x["status"] == 0 for q in ss for x in q["levels"])), solves=int(sum(len(q["levels"]) for q in ss)),
                        kept=float(np.mean([q["n_kept"] for q in ss])))
    cache = M.load_cache()
    anch = {}
    for m_ in (10, 20, 30):
        vals = [v for k, v in cache.items() if dict(k).get("M") == m_ and set(dict(k)) == {"M", "seed"}]
        anch[m_] = dict(t_lat=float(np.mean([v["t_lat"] for v in vals])), t_en=float(np.mean([v["t_en"] for v in vals])), t_en_max=float(max(v["t_en"] for v in vals)),
                        cand_mean=float(np.mean([len(v["info"]["candidates"]) for v in vals])), cand_max=int(max(len(v["info"]["candidates"]) for v in vals)))
    return tab, conv, anch


def discrete(out):
    """Round the ACE-SQP frequencies up to L equally spaced DVFS steps and report the HV loss."""
    rep = {}
    for L in (4, 8, 16):
        loss = []
        for o in out:
            if o["name"] != M.PROP:
                continue
            sc = Scenario(**o["kw"])
            X = o["X"].copy()
            u = X[:, 3 * sc.M:]
            X[:, 3 * sc.M:] = np.ceil(u * (L - 1) - 1e-12) / (L - 1)
            F0, F1 = M.nondom(objectives(o["X"], sc)), M.nondom(objectives(X, sc))
            lo, hi = F0.min(0), F0.max(0)
            loss.append(HV_((F0 - lo) / (hi - lo)) - HV_((F1 - lo) / (hi - lo)))
        rep[L] = dict(mean=float(np.mean(loss)), max=float(np.max(loss)))
    return rep


if __name__ == "__main__":
    M.init_worker()
    out = load("e1")
    info, df, sm = e1_tables(out)
    base_figs(out, info)
    info["e3"] = e3(load("e3"))
    tab, conv, anch = e4(load("e4"), out)
    info["e4"], info["e4_conv"], info["anchors"] = tab, conv, anch
    abl = pd.DataFrame(load("abl"))
    rows = []
    for s, g in abl.groupby("s"):
        ace = [o for o in out if o["name"] == M.PROP and o["kw"]["seed"] == s][0]
        ref = M.nondom(np.vstack(list(g.F) + [ace["F"]]))
        ev, _ = metric_fns(ref)
        rows.append(dict(s=s, label=M.PROP, HV=ev(ace["F"])[0], IGD=ev(ace["F"])[1], evals=ace["evals"], conv=sum(x["status"] == 0 for x in ace["stats"]["levels"]), levels=len(ace["stats"]["levels"]), kept=ace["stats"]["n_kept"]))
        for _, r in g.iterrows():
            h, i_ = ev(r.F)
            rows.append(dict(s=s, label=r.label, HV=h, IGD=i_, evals=r.evals, conv=r.conv, levels=r.levels, kept=r.kept))
    a = pd.DataFrame(rows)
    info["ablation"] = a.groupby("label").agg(HV=("HV", "mean"), IGD=("IGD", "mean"), evals=("evals", "mean"), conv=("conv", "sum"), levels=("levels", "sum"), kept=("kept", "mean")).reset_index().to_dict("records")
    g = pd.DataFrame(load("glob"))
    g["gap"] = 100 * (g.ladder / g.best - 1)
    info["multistart"] = dict(cases=len(g), mean_gap=float(g.gap.mean()), max_gap=float(g.gap.max()), min_gap=float(g.gap.min()), n_ok=float(g.n_ok.mean()), spread=float(g.spread.mean()),
                              worse_1pct=int((g.gap > 1).sum()), by_level=g.groupby("k").gap.agg(["mean", "max"]).reset_index().to_dict("records"))
    info["discrete"] = discrete(out)
    json.dump(info, open(os.path.join(R, "summary_v7.json"), "w"), indent=1, default=float)
    print(json.dumps({k: info[k] for k in ("table", "friedman", "within_sd", "ace", "budget", "base_tradeoff", "sr_at_25", "ablation", "multistart", "discrete", "e3", "e4", "e4_conv", "anchors")}, indent=0, default=float)[:12000])

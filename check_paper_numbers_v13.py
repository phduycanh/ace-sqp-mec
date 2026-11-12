"""
Checks the numbers that the fourth revision of the paper (v13) states in Sections VI-B and VI-I (restart variants,
E1 diagnosis, distance of the GA of Elkawkagy et al. to T*) against results/summary_v13.json, timing_v13.json,
ga9_latency.json and the raw files of the variants (results/raw_v13_*.pkl, needed only for the front sizes).
Usage: python check_paper_numbers_v13.py   (prints one line per statement and "ALL OK" at the end)
"""
import json, os, pickle
import numpy as np

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
S = json.load(open(os.path.join(R, "summary_v13.json")))
TM = json.load(open(os.path.join(R, "timing_v13.json")))
G = json.load(open(os.path.join(R, "ga9_latency.json")))
ok = []


def chk(text, cond):
    ok.append(bool(cond))
    print("OK  " if cond else "FAIL", text)


r = lambda b, m: S[b][m]
SETS = [("e1", "10"), ("stress", "10"), ("path", "10"), ("e4", "20"), ("e4", "30"), ("e4b", "50"), ("fresh50", "50")]
# Section VI-B (E1)
h = S["e1_hvlow_vs_nbi"]
chk("E1: NBI-SQP higher HV_2T* in 25/30, by 0.002 (median), at most 0.004", h["n_nbi_higher"] == 25 and
    round(h["nbi_higher_median"], 3) == 0.002 and round(h["nbi_higher_max"], 3) == 0.004)
chk("E1: lower by 0.008-0.028 in the other five", round(h["nbi_lower_min"], 3) == 0.008 and round(h["nbi_lower_max"], 3) == 0.028)
e = S["e1_diagnosis"]
chk("E1: five scenarios = 84 % of the summed HV advantage", round(100 * e["share_of_total_from_top5"]) == 84)
chk("E1: Spearman rho = 0.68, p < 1e-4", round(e["spearman_nbi_vs_noact"]["rho"], 2) == 0.68 and e["spearman_nbi_vs_noact"]["p"] < 1e-4)
chk("E1: four of the five are the only scenarios with an idle UD", sum(t["idle"] for t in e["top5"]) == 4)
chk("E1: base scenario 2026 among the five where NBI-SQP has the higher HV", 2026 in e["nbi_better"] and len(e["nbi_better"]) == 5)
chk(f"E1: GA of [9] on average 13.7 % above T* (6.2-27.9 %)", f"{G['mean']:.1f}" == "13.7" and f"{G['min']:.1f}" == "6.2"
    and f"{G['max']:.1f}" == "27.9")
# Section VI-I (restart versus scalarization)
g = [r(b, m)["pairs"]["HV"]["NBI+act better than NBI"] for b, m in SETS]
chk("restart improves NBI-SQP on every set, in all scenarios of five sets",
    all(x["gain"] > 0 for x in g) and sum(x["n"] == r(b, m)["n"] for x, (b, m) in zip(g, SETS)) == 5)
p = [r(b, m)["vs_ace"]["HV"]["NBI-SQP+act"]["p_holm"] for b, m in [("e1", "10"), ("path", "10"), ("e4", "20"), ("e4", "30")]]
chk("NBI-SQP+act vs ACE-SQP on E1, path loss, M = 20, 30: Holm p >= 0.10", min(p) >= 0.10)
s1, s2 = r("e1", "10")["pairs"]["HVlow"]["NBI+act strictly higher than ACE"], r("e4", "20")["pairs"]["HVlow"]["NBI+act strictly higher than ACE"]
chk("HV_2T*: NBI-SQP+act higher in all 30 E1 scenarios and all ten with M = 20", s1["n"] == 30 and s2["n"] == 10)
chk("HV_2T* on E1: 1.0216 against 1.0198", round(r("e1", "10")["means"]["HVlow"]["NBI-SQP+act"], 4) == 1.0216
    and round(r("e1", "10")["means"]["HVlow"]["ACE-SQP"], 4) == 1.0198)
st = r("stress", "10")
d = st["means"]["HVlow"]["ACE-SQP"] - st["means"]["HVlow"]["NBI-SQP+act"]
chk("heterogeneous: ACE-SQP higher HV in 10/10, mean HV_2T* +0.024 (5 of 10)", st["vs_ace"]["HV"]["NBI-SQP+act"]["ace_better"] == 10
    and round(d, 3) == 0.024 and st["vs_ace"]["HVlow"]["NBI-SQP+act"]["ace_better"] == 5)
x = r("e4b", "50")["vs_ace"]["HV"]["NBI-SQP+act"]
chk("M = 50 (original): 8 of 10, p = 0.02", x["ace_better"] == 8 and round(x["p_holm"], 2) == 0.02)
a = [r(b, m)["pairs"]["HV"]["activation better than random restart"] for b, m in SETS]
chk("activated vs random point: 28/30 (E1), 10/10 (heterogeneous), at most 2e-4 on average",
    a[0]["n"] == 28 and a[1]["n"] == 10 and max(abs(z["diff"]) for z in a) <= 2e-4)
gain = [st["means"]["HV"][k] - st["means"]["HV"]["ACE-SQP w/o activation"] for k in ("ACE-SQP", "ACE-SQP (random restart)")]
chk("either restart gains 0.011 over no restart (heterogeneous)", all(round(v, 3) == 0.011 for v in gain))
chk("no HV_2T* difference favors the activated point", min(r(b, m)["pairs"]["HVlow"]["activation better than random restart"]["p"] for b, m in SETS) > 0.05)
rr = [TM[f"M{m}|ACE-SQP (random restart)"]["median"] / TM[f"M{m}|ACE-SQP"]["median"] for m in (10, 20, 30, 50)]
chk(f"random point: 0.65-1.00 times the run time ({min(rr):.2f}-{max(rr):.2f})", f"{min(rr):.2f}" == "0.65" and f"{max(rr):.2f}" == "1.00")
chk("random second starts lower the HV of NBI-SQP on every set",
    all(r(b, m)["means"]["HV"]["NBI-SQP+rand"] < r(b, m)["means"]["HV"]["NBI-SQP"] for b, m in SETS))
raw = [os.path.join(R, f"raw_v13_{b}.pkl") for b in ("e1", "stress", "path", "e4", "e4b", "fresh50")]
if all(os.path.exists(f) for f in raw):
    sz = []
    for f in raw:
        recs = [o for o in pickle.load(open(f, "rb")) if o["name"] == "NBI-SQP+rand"]
        for m in sorted({o["kw"]["M"] for o in recs}):
            sz.append(np.mean([len(o["F"]) for o in recs if o["kw"]["M"] == m]))
    chk(f"front sizes with random second starts: 75-95 points ({min(sz):.1f}-{max(sz):.1f})", round(min(sz)) == 75 and round(max(sz)) == 95)
else:
    print("skip front sizes (raw_v13_*.pkl not present; run experiments_v13.py or download the raw records)")
print("ALL OK" if all(ok) else f"{ok.count(False)} FAILED")

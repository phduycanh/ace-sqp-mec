"""
V12 (reviewer item M9): a single operating point at (1 + delta) T* solved directly from the latency anchor, compared
with the point of the ACE-SQP front within the same latency budget. Output: results/summary_v12.json
"""
import json, os, pickle
import numpy as np
import pandas as pd
import methods as M

R = M.RESULTS
raw = pickle.load(open(os.path.join(R, "raw_single.pkl"), "rb"))
rows = []
for o in raw:
    for p in o["pts"]:
        rows.append(dict(M=o["kw"]["M"], seed=o["kw"]["seed"], delta=p["delta"], act=p["act"],
                         excess=100 * (p["E"] / p["ladder"] - 1), evals=p["evals"], conv=p["status"] == 0,
                         save_single=100 * (1 - p["E"] / o["E0"]), save_ladder=100 * (1 - p["ladder"] / o["E0"])))
df = pd.DataFrame(rows)
g = df.groupby(["M", "delta", "act"]).agg(excess_mean=("excess", "mean"), excess_max=("excess", "max"),
                                          excess_min=("excess", "min"), over_1pct=("excess", lambda x: int((x > 1).sum())),
                                          n=("excess", "size"), evals=("evals", "mean"), conv=("conv", "mean"),
                                          save_single=("save_single", "mean"), save_ladder=("save_ladder", "mean")).reset_index()
out = dict(single=g.to_dict("records"))
tm = json.load(open(os.path.join(R, "timing.json")))
out["timing_single"] = tm.get("single")
out["timing_ace"] = {k: v["median"] for k, v in tm["methods"].items() if k.endswith("|" + M.PROP)}
out["anchors"] = {k: v for k, v in tm.items() if k.startswith("anchor_median")}
json.dump(out, open(os.path.join(R, "summary_v12.json"), "w"), indent=1, default=float)
pd.set_option("display.width", 200)
print(g.round(3).to_string())
print(json.dumps(out["timing_single"], indent=0)[:1500])

"""Twin-differenced fast-modulation probability (mech_twin.py) across states: healthy TDBRAIN eyes open against
closed, and sleep stages (Sleep-EDF, ANPHY), with its relation to the posterior fold distance.
usage: python scripts/mech_twin_summary.py"""
import json, numpy as np, pandas as pd
from scipy.stats import wilcoxon, spearmanr
from ampint.io_tdbrain import load_meta

out = {}
meta = load_meta(); m1 = meta[meta.sessID == 1].drop_duplicates("sub").set_index("sub")
T = pd.read_csv("results/mechtw_tdbrain.csv")
T = T[(T.ses == "ses-1") & (T["sub"].map(m1.indication) == "HEALTHY") & (T["sub"].map(m1.age) >= 18)]
s = T.groupby(["sub", "task"]).p_fast.mean().unstack().dropna()
out["TDBRAIN healthy"] = dict(n=int(len(s)), EC=float(s.restEC.mean()), EO=float(s.restEO.mean()),
                              EO_minus_EC=float((s.restEO - s.restEC).mean()), p=float(wilcoxon(s.restEO, s.restEC).pvalue),
                              rho_pfast_delta=float(spearmanr(T.p_fast, T.delta)[0]))
for name in ("sleep_alpha", "anphy_alpha", "sleep_sigma", "anphy_sigma"):
    d = pd.read_csv(f"results/mechtw_{name}.csv")
    d["subj"] = d.rec.str[:5] if name.startswith("sleep") else d.rec
    g = d.groupby(["subj", "stage"]).p_fast.mean().unstack()
    r = {st: float(g[st].mean()) for st in ("W", "N1", "N2", "N3", "REM") if st in g}
    for st in ("N1", "N2", "N3", "REM"):
        x = g[["W", st]].dropna()
        r[f"{st}-W"] = dict(n=int(len(x)), diff=float((x[st] - x.W).mean()), p=float(wilcoxon(x[st], x.W).pvalue))
    r["rho_pfast_delta"] = float(spearmanr(d.p_fast, d.delta)[0])
    out[name] = r
json.dump(out, open("results/mech_twin_summary.json", "w"), indent=1)
print(json.dumps(out, indent=1))

"""Monte Carlo stability of the predictive burst check. ppc_bursts.py was rerun with 16 realizations per channel
recording (results/bursts_ppc_ns16.csv; realizations 0-3 are those of the main analysis). For each statistic and
condition: Spearman correlation between observed and simulated recording-level statistics with 4 and with 16
realizations, and the Monte Carlo standard error of the 4-realization recording mean relative to the spread of the
simulated means across recordings.
usage: python scripts/ppc_stability.py"""
import glob, json, numpy as np, pandas as pd
from scipy.stats import spearmanr

STATS = ["b_frac", "b2_frac", "b_dur_mean", "b_dur_p90", "b2_dur_mean", "b_rate"]
key = ["sub", "task", "ch"]
b = pd.concat([pd.read_csv(f) for f in glob.glob("results/bursts_tdbrain/*.csv") if open(f).readline().strip()])
obs = b[b.kind == "data"].set_index(key)[STATS].groupby(level=["sub", "task"]).mean()
S = pd.read_csv("results/bursts_ppc_ns16.csv")
out = {}
for task in ("restEC", "restEO"):
    s = S[S.task == task]
    m4 = s[s.seed < 4].groupby(key)[STATS].mean().groupby(level=["sub", "task"]).mean()
    m16 = s.groupby(key)[STATS].mean().groupby(level=["sub", "task"]).mean()
    o = obs.xs(task, level="task", drop_level=False)
    idx = o.index.intersection(m4.index).intersection(m16.index)
    for st in STATS:
        # Monte Carlo SE of a 4-realization channel mean, averaged over the 4 channels of a recording
        sd_ch = s.groupby(key)[st].std(ddof=1)
        se4 = (sd_ch / np.sqrt(4)).groupby(level=["sub", "task"]).apply(lambda v: np.sqrt(np.nanmean(v ** 2) / v.notna().sum()))
        out[f"{task}|{st}"] = dict(n=len(idx), rho_4=float(spearmanr(o.loc[idx, st], m4.loc[idx, st], nan_policy="omit")[0]),
                                   rho_16=float(spearmanr(o.loc[idx, st], m16.loc[idx, st], nan_policy="omit")[0]),
                                   rho_4_vs_16=float(spearmanr(m4.loc[idx, st], m16.loc[idx, st], nan_policy="omit")[0]),
                                   mc_se_over_between_sd=float(np.nanmedian(se4.loc[idx]) / np.nanstd(m16.loc[idx, st])))
json.dump(out, open("results/ppc_stability.json", "w"), indent=1)
print(pd.DataFrame(out).T.round(3).to_string())

"""Cross-session burst prediction (TDBRAIN, all participants with two sessions): session-2 burst statistics
predicted from session-1 data only, by (i) the normal form at the session-1 posterior medians, (ii) the
session-1 linear twin and (iii) the session-1 observed value (test-retest ceiling). Recording level = mean
over O1, Oz, O2, Pz."""
import glob, json, numpy as np, pandas as pd
from scipy.stats import spearmanr, wilcoxon

STATS = ["b_dur_mean", "b_dur_p90", "b_frac", "b2_dur_mean", "b2_frac", "b_rate"]
b = pd.concat([pd.read_csv(f) for f in glob.glob("results/bursts_retest/*.csv") if open(f).readline().strip()])
key = ["sub", "task", "ch"]
obs2 = b[(b.kind == "data") & (b.ses == "ses-2")].set_index(key)[STATS]
obs1 = b[(b.kind == "data") & (b.ses == "ses-1")].set_index(key)[STATS]
twin1 = b[(b.kind == "twin") & (b.ses == "ses-1")].set_index(key)[STATS]
pred = pd.read_csv("results/bursts_ppc_retest.csv").groupby(key)[STATS].mean()
idx = obs2.index.intersection(obs1.index).intersection(twin1.index).intersection(pred.index)
L = pd.concat({"obs2": obs2.loc[idx], "obs1": obs1.loc[idx], "twin1": twin1.loc[idx], "pred1": pred.loc[idx]}, axis=1)
R = L.groupby(level=["sub", "task"]).mean()
out = dict(n_participants={t: int((R.index.get_level_values("task") == t).sum()) for t in ("restEC", "restEO")})
for task in ("restEC", "restEO"):
    r = R.xs(task, level="task")
    for s in STATS:
        o2, o1, tw, p = r[("obs2", s)], r[("obs1", s)], r[("twin1", s)], r[("pred1", s)]
        ok = o2.notna() & o1.notna() & tw.notna() & p.notna()
        o2, o1, tw, p = o2[ok], o1[ok], tw[ok], p[ok]
        ep, et = (o2 - p).abs(), (o2 - tw).abs()
        # does the session-1 prediction carry information about session 2 beyond the session-1 twin?
        X = np.column_stack([np.ones(ok.sum()), tw.rank(), p.rank()]); yv = o2.rank().values
        beta, res, *_ = np.linalg.lstsq(X, yv, rcond=None)
        sig2 = np.sum((yv - X @ beta) ** 2) / (len(yv) - 3); se = np.sqrt(sig2 * np.linalg.inv(X.T @ X)[2, 2])
        out[f"{task}|{s}"] = dict(n=int(ok.sum()), rho_pred1=float(spearmanr(o2, p)[0]), rho_twin1=float(spearmanr(o2, tw)[0]),
                                  rho_obs1=float(spearmanr(o2, o1)[0]), mae_pred1=float(ep.mean()), mae_twin1=float(et.mean()),
                                  frac_pred_closer=float((ep < et).mean()), p_mae=float(wilcoxon(ep, et).pvalue),
                                  t_pred_beyond_twin=float(beta[2] / se))
json.dump(out, open("results/bursts_retest_summary.json", "w"), indent=1)
pd.set_option("display.width", 220)
print(pd.DataFrame({k: v for k, v in out.items() if isinstance(v, dict) and "n" in v}).T.round(3).to_string())
print(out["n_participants"])

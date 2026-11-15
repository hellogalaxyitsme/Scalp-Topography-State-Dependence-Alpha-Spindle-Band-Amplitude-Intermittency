"""Posterior predictive check on alpha bursts (healthy TDBRAIN, first session): observed burst statistics against
the normal-form prediction (posterior median coordinates, mean over simulation seeds) and against linear twins.
Recording level = mean over the channels with twins (O1, Oz, O2, Pz)."""
import glob, json, numpy as np, pandas as pd
from scipy.stats import spearmanr, wilcoxon
from ampint.nf import fold_delta

STATS = ["b_dur_mean", "b_dur_p90", "b_frac", "b_rate", "b_gap_mean", "b2_dur_mean", "b2_frac"]

b = pd.concat([pd.read_csv(f) for f in glob.glob("results/bursts_tdbrain/*.csv") if open(f).readline().strip()])
obs = b[b.kind == "data"].set_index(["sub", "task", "ch"])[STATS]
twin = b[b.kind == "twin"].set_index(["sub", "task", "ch"])[STATS]
pred = pd.read_csv("results/bursts_ppc.csv").groupby(["sub", "task", "ch"])[STATS].mean()
post = pd.read_csv("results/post_tdbrain.csv", usecols=["sub", "ses", "task", "ch", "kind", "th_mu_q50", "th_c_q50", "log_t0_q50"])
post["cov_pct"] = pd.read_csv("results/coverage_tdbrain.csv").cov_pct.values
post = post[(post.kind == "data") & (post.ses == "ses-1")].set_index(["sub", "task", "ch"])
post["delta"] = fold_delta(post.th_mu_q50, post.th_c_q50)
idx = obs.index.intersection(pred.index).intersection(twin.index)
L = pd.concat({"obs": obs.loc[idx], "twin": twin.loc[idx], "pred": pred.loc[idx]}, axis=1)
L[("post", "delta")] = post.reindex(idx)["delta"].values
L[("post", "log_t0")] = post.reindex(idx)["log_t0_q50"].values
L[("post", "covered")] = (post.reindex(idx)["cov_pct"].values < 95).astype(float)
R = L.groupby(level=["sub", "task"]).mean()
R.columns = ["|".join(c) for c in R.columns]
R.to_csv("results/bursts_reclevel.csv")

out = dict(n_channel_recordings=int(len(L)), n_recordings={t: int((R.index.get_level_values("task") == t).sum()) for t in ("restEC", "restEO")})
for task in ("restEC", "restEO"):
    r = R.xs(task, level="task")
    for s in STATS:
        o, tw, p = r[f"obs|{s}"], r[f"twin|{s}"], r[f"pred|{s}"]
        ok = o.notna() & tw.notna() & p.notna()
        o, tw, p, dl = o[ok], tw[ok], p[ok], r["post|delta"][ok]
        e_p, e_t = (o - p).abs(), (o - tw).abs()
        out[f"{task}|{s}"] = dict(
            n=int(ok.sum()), median_obs=float(o.median()), median_twin=float(tw.median()), median_pred=float(p.median()),
            p_obs_vs_twin=float(wilcoxon(o, tw).pvalue), rho_obs_pred=float(spearmanr(o, p)[0]),
            rho_obs_twin=float(spearmanr(o, tw)[0]), mae_pred=float(e_p.mean()), mae_twin=float(e_t.mean()),
            frac_pred_closer=float((e_p < e_t).mean()), p_mae=float(wilcoxon(e_p, e_t).pvalue),
            rho_excess_obs_pred=float(spearmanr(o - tw, p - tw)[0]), rho_excess_delta=float(spearmanr(o - tw, dl)[0]))
# channel level, by simulated support: is the prediction worse where the summaries lie outside the bank?
for cov, lab in ((1.0, "inside support"), (0.0, "outside support")):
    c = L[L[("post", "covered")] == cov]
    for s in ("b_dur_mean", "b2_frac"):
        o, p, tw = c[("obs", s)], c[("pred", s)], c[("twin", s)]
        ok = o.notna() & p.notna() & tw.notna()
        out[f"channel {lab}|{s}"] = dict(n=int(ok.sum()), rho_obs_pred=float(spearmanr(o[ok], p[ok])[0]),
                                        rho_obs_twin=float(spearmanr(o[ok], tw[ok])[0]),
                                        frac_pred_closer=float(((o - p).abs() < (o - tw).abs())[ok].mean()),
                                        median_rel_err_pred=float(((p - o) / o)[ok].median()))
# eyes open - closed: observed change against predicted and twin change
w = R.reset_index()
for s in STATS:
    pv = w.pivot_table(index="sub", columns="task", values=[f"obs|{s}", f"pred|{s}", f"twin|{s}"]).dropna()
    d = {k: pv[(f"{k}|{s}", "restEO")] - pv[(f"{k}|{s}", "restEC")] for k in ("obs", "pred", "twin")}
    out[f"EO-EC|{s}"] = dict(n=int(len(pv)), mean_obs=float(d["obs"].mean()), mean_pred=float(d["pred"].mean()),
                            mean_twin=float(d["twin"].mean()), rho_obs_pred=float(spearmanr(d["obs"], d["pred"])[0]),
                            rho_obs_twin=float(spearmanr(d["obs"], d["twin"])[0]))
json.dump(out, open("results/bursts_summary.json", "w"), indent=1)
pd.set_option("display.width", 250)
print(pd.DataFrame({k: v for k, v in out.items() if isinstance(v, dict)}).T.round(3).to_string())

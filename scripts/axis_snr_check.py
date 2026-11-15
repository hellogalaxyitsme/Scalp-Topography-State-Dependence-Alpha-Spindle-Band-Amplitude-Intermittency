"""Can a change in signal-to-noise ratio alone produce the state-axis displacement? On held-out simulations in
the healthy region, the estimation error of (mu, c) is regressed on the true SNR and 1/f exponent; the implied
error displacement for the observed eyes-open change in estimated SNR is compared with the observed displacement."""
import json, numpy as np, pandas as pd

ho = pd.read_csv("results/sbi_alpha_full_heldout.csv")
reg = ho[ho.th_mu.between(-7, -3) & ho.th_c.between(1.5, 5.5)].copy()
X = np.column_stack([np.ones(len(reg)), reg.th_snr_db, reg.th_chi, reg.th_mu, reg.th_c])
out = dict(n=int(len(reg)))
slope = {}
for k in ("mu", "c"):
    e = reg[f"th_{k}_q50"].values - reg[f"th_{k}"].values
    b = np.linalg.lstsq(X, e, rcond=None)[0]
    slope[k] = float(b[1])
    out[f"err_{k}_per_dB"] = float(b[1])
R = pd.read_csv("results/nf_tdbrain_reclevel.csv")
post = pd.read_csv("results/post_tdbrain.csv", usecols=["sub", "ses", "task", "ch", "kind", "th_snr_db_hat"])
post = post[(post.kind == "data") & (post.ses == "ses-1") & post.ch.isin(["O1", "Oz", "O2", "Pz", "P3", "P4"])]
snr = post.groupby(["sub", "task"]).th_snr_db_hat.mean().unstack()
H = R[(R.group == "HEALTHY") & (R.task == "restEC")]["sub"]
d_snr = (snr.loc[snr.index.isin(H), "restEO"] - snr.loc[snr.index.isin(H), "restEC"]).dropna()
out["eo_minus_ec_snr_db_mean"] = float(d_snr.mean())
art = np.array([slope["mu"], slope["c"]]) * d_snr.mean()
disp = json.load(open("results/displacements.json"))["eyes open - closed (healthy)"]
obs = np.array([disp["d_mu"], disp["d_c"]])
out.update(snr_artifact_d_mu=float(art[0]), snr_artifact_d_c=float(art[1]), observed_d_mu=float(obs[0]), observed_d_c=float(obs[1]),
           artifact_over_observed_length=float(np.linalg.norm(art) / np.linalg.norm(obs)),
           cos_artifact_observed=float(art @ obs / (np.linalg.norm(art) * np.linalg.norm(obs) + 1e-12)))
json.dump(out, open("results/axis_snr_check.json", "w"), indent=1)
print(json.dumps(out, indent=1))

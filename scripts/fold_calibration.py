"""Calibration of the plug-in fold distance on held-out simulations: distribution of the estimate for rhythms
truly near the fold (|delta| < 0.5) and deep in the focus region (delta < -3), the probability of an estimate at
least as high as the observed values under each, and the likelihood ratio, for the sigma (spindle) and alpha
settings. Observed values are read from the analysis outputs."""
import json, numpy as np, pandas as pd
from ampint.nf import fold_delta

obs = {}
F = json.load(open("results/fold_stats.json"))
obs["sigma250"] = {"Sleep-EDF Fpz-Cz N2": F["sleep_sigma_fpz"]["median_delta_data"]["N2"],
                   "Sleep-EDF Fpz-Cz N2 twin": F["sleep_sigma_fpz"]["median_delta_twin"]["N2"],
                   "Sleep-EDF Fpz-Cz W": F["sleep_sigma_fpz"]["median_delta_data"]["W"]}
A = json.load(open("results/anphy_summary.json"))["all"]
for k, lab in (("data_twin_chunks|scalp mean", "ANPHY scalp N2"), ("data_twin_chunks|FPZ-CZ", "ANPHY Fpz-Cz N2"),
               ("twin|scalp mean", "ANPHY scalp N2 twin")):
    obs["sigma250"][lab] = A[k]["N2"]["delta"]
T = F["tdbrain"]
obs["alpha"] = {f"TDBRAIN healthy {t} {s}": T[t][f"median_{s}"] for t in ("restEC", "restEO") for s in ("data", "twin")}
out = {}
for cfg, o in obs.items():
    H = pd.read_csv(f"results/sbi_{cfg}_full_heldout.csv", usecols=["th_mu", "th_c", "th_mu_q50", "th_c_q50"])
    dt, dh = fold_delta(H.th_mu, H.th_c), fold_delta(H.th_mu_q50, H.th_c_q50)
    near, deep = np.abs(dt) < 0.5, dt < -3
    r = dict(n_near=int(near.sum()), n_deep=int(deep.sum()),
             estimate_quantiles_near={q: float(np.quantile(dh[near], q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
             estimate_quantiles_deep={q: float(np.quantile(dh[deep], q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)})
    for lab, x in o.items():
        pn, pd_ = float(np.mean(dh[near] >= x)), float(np.mean(dh[deep] >= x))
        win = np.abs(dh - x) < 0.25          # local likelihood ratio of near vs deep truth at the observed estimate
        r[lab] = dict(observed=float(x), p_ge_if_near=pn, p_ge_if_deep=pd_,
                      local_lr_near_vs_deep=float((np.mean(win[near]) + 1e-6) / (np.mean(win[deep]) + 1e-6)),
                      median_true_delta_given_estimate=float(np.median(dt[win])) if win.sum() > 20 else None)
    out[cfg] = r
json.dump(out, open("results/fold_calibration.json", "w"), indent=1)
print(json.dumps(out, indent=1))

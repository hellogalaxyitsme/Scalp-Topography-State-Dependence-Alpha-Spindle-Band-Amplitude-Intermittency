"""Directions in which known perturbations move the posterior medians, relative to the state axis: fast hidden
modulation of the Hopf parameter and source mixing (sim_alternatives.py), slow modulation (modulated bank) and a
change in SNR (axis_snr_check.py). Shifts are differences of mean estimates against the matching unperturbed
simulations."""
import json, numpy as np, pandas as pd

A = json.load(open("results/state_axis.json"))["axis"]; u = np.array([A["u_mu"], A["u_c"]])
R = pd.read_csv("results/sim_alternatives.csv")
out = dict(axis=A, fast_modulation={}, mixing={})
for cfg in ("alpha", "sigma250"):
    d = R[(R.case == "drive") & (R.cfg == cfg)]
    base = d[d.sd == 0][["th_mu_q50", "th_c_q50"]].mean().values
    for (sd, tau), g in d[d.sd > 0].groupby(["sd", "tau"]):
        v = g[["th_mu_q50", "th_c_q50"]].mean().values - base
        out["fast_modulation"][f"{cfg} sd={sd} tau={tau}s"] = dict(d_mu=float(v[0]), d_c=float(v[1]), along=float(v @ u),
                                                                    cos_axis=float(v @ u / np.linalg.norm(v)))
m = R[R.case == "mix"]
for dt in sorted(m.true_delta.unique()):
    b = m[(m.true_delta == dt) & (m.n_src == 1)][["th_mu_q50", "th_c_q50"]].mean().values
    for (n, df), g in m[(m.true_delta == dt) & (m.n_src > 1)].groupby(["n_src", "df_hz"]):
        v = g[["th_mu_q50", "th_c_q50"]].mean().values - b
        out["mixing"][f"true delta {dt}, {int(n)} sources, spacing {df} Hz"] = dict(d_mu=float(v[0]), d_c=float(v[1]), along=float(v @ u),
                                                                                     cos_axis=float(v @ u / np.linalg.norm(v)))
mod = pd.read_csv("results/sbi_alpha_modulated_eval.csv")
cols = [c for c in ("th_mu", "th_c", "th_mu_q50", "th_c_q50") if c in mod.columns]
if len(cols) == 4:
    v = np.array([(mod.th_mu_q50 - mod.th_mu).mean(), (mod.th_c_q50 - mod.th_c).mean()])
    out["slow_modulation (bank, tau 3-30 s)"] = dict(d_mu=float(v[0]), d_c=float(v[1]), along=float(v @ u), cos_axis=float(v @ u / np.linalg.norm(v)))
s = json.load(open("results/axis_snr_check.json"))
v = np.array([s["snr_artifact_d_mu"], s["snr_artifact_d_c"]])
out["SNR change of the eyes-open size"] = dict(d_mu=float(v[0]), d_c=float(v[1]), along=float(v @ u), cos_axis=float(v @ u / np.linalg.norm(v)))
json.dump(out, open("results/axis_perturbations.json", "w"), indent=1)
for k in ("fast_modulation", "mixing"):
    cs = [x["cos_axis"] for x in out[k].values()]
    print(k, f"cos with axis: {min(cs):+.2f} to {max(cs):+.2f}")
for k in ("slow_modulation (bank, tau 3-30 s)", "SNR change of the eyes-open size"):
    if k in out:
        print(k, {kk: round(vv, 3) for kk, vv in out[k].items()})

"""Signed distance to the fold of limit cycles, delta = mu + c^2/4 (c>0; <0 inside the focus region, 0 on the fold).
For c<=0 the fold does not exist; delta := mu (distance to the supercritical Hopf line).
Data and IAAFT twins are compared on the same channels (TDBRAIN: O1, Oz, O2, Pz) and the same sleep chunks.
Sensitivity: rows whose summary vectors lie outside the simulated support (coverage_rows.py) are excluded."""
import json, numpy as np, pandas as pd
from scipy.stats import wilcoxon

TWIN_CH = ["O1", "Oz", "O2", "Pz"]


def delta(mu, c):
    return np.where(c > 0, mu + c * c / 4, mu)


def tdbrain(P, covered):
    out = {}
    d = P[P.cov_pct < 95] if covered else P
    for task in ["restEC", "restEO"]:
        s = d[d.task == task].groupby(["sub", "src"]).delta.mean().unstack().dropna()
        w = wilcoxon(s.data, s.twin)
        out[task] = dict(n=int(len(s)), median_data=float(s.data.median()), median_twin=float(s.twin.median()),
                         median_diff=float((s.data - s.twin).median()), p=float(w.pvalue),
                         frac_within_1_of_fold_data=float((s.data.abs() < 1).mean()), frac_within_1_twin=float((s.twin.abs() < 1).mean()))
    return out


def sleep(S, covered):
    d = S[S.cov_pct < 95] if covered else S
    d = d[d.chunk % 4 == 0]                      # chunks with twins
    ss = d[d.kind == "data"].groupby(["subj", "stage"]).delta.mean().unstack()
    st = d[d.kind == "iaaft0"].groupby(["subj", "stage"]).delta.mean().unstack()
    return dict(median_delta_data=ss.median().round(3).to_dict(), median_delta_twin=st.median().round(3).to_dict(),
                n_subjects=ss.notna().sum().to_dict())


P = pd.read_csv("results/post_tdbrain.csv", usecols=["sub", "ses", "task", "ch", "kind", "th_mu_q50", "th_c_q50"])
P["cov_pct"] = pd.read_csv("results/coverage_tdbrain.csv").cov_pct.values
R = pd.read_csv("results/nf_tdbrain_reclevel.csv")
hs = set(R[R.group == "HEALTHY"]["sub"])
P = P[P.ch.isin(TWIN_CH) & P["sub"].isin(hs) & (P.ses == "ses-1")].copy()
P["delta"] = delta(P.th_mu_q50.values, P.th_c_q50.values)
P["src"] = np.where(P.kind == "data", "data", "twin")
S = pd.read_csv("results/post_sleep_sigma.csv", usecols=["rec", "ch", "stage", "chunk", "kind", "th_mu_q50", "th_c_q50"])
S["cov_pct"] = pd.read_csv("results/coverage_sleep_sigma.csv").cov_pct.values
S = S[S.ch == "EEG Fpz-Cz"].copy()
S["delta"] = delta(S.th_mu_q50.values, S.th_c_q50.values); S["subj"] = S.rec.str[:5]
out = dict(tdbrain=tdbrain(P, False), tdbrain_covered=tdbrain(P, True), sleep_sigma_fpz=sleep(S, False), sleep_sigma_fpz_covered=sleep(S, True))
json.dump(out, open("results/fold_stats.json", "w"), indent=1)
print(json.dumps(out, indent=1))

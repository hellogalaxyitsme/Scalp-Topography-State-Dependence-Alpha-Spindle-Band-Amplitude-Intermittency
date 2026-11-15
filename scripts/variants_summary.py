"""Recorded reference against EOG regression and surface Laplacian (healthy TDBRAIN adults, first session):
fold distance of the EEG and its linear twins on O1, Oz, O2, Pz; eyes-open minus eyes-closed effects and their
scalp maps (correlation with the recorded-reference map); regime fractions. ANPHY: recorded (average) reference
against surface Laplacian for the N2 spindle results."""
import json, numpy as np, pandas as pd
from scipy.stats import wilcoxon, pearsonr
from ampint.nf import fold_delta
from ampint.io_tdbrain import load_meta

TW = ["O1", "Oz", "O2", "Pz"]
meta = load_meta()
hs = set(meta[(meta.sessID == 1) & (meta.indication == "HEALTHY") & (meta.age >= 18) & meta.gender.notna()]["sub"])
out = {}
topo0 = json.load(open("results/topo_tdbrain.json"))["all"]
for var in ("", "eog", "csd"):
    sfx = f"_{var}" if var else ""
    P = pd.read_csv(f"results/post_tdbrain{sfx}.csv", usecols=["sub", "ses", "task", "ch", "kind", "th_mu_q50", "th_c_q50",
                                                               "log_t0_q50", "p_focus", "p_bistable", "p_oscillatory"])
    P = P[P["sub"].isin(hs) & (P.ses == "ses-1")].copy()
    P["delta"] = fold_delta(P.th_mu_q50, P.th_c_q50); P["src"] = np.where(P.kind == "data", "data", "twin")
    r = {}
    for task in ("restEC", "restEO"):
        s = P[P.ch.isin(TW) & (P.task == task)].groupby(["sub", "src"]).delta.mean().unstack().dropna()
        reg = P[(P.kind == "data") & P.ch.isin(TW + ["P3", "P4"]) & (P.task == task)][["p_focus", "p_bistable", "p_oscillatory"]].idxmax(1)
        r[task] = dict(n=int(len(s)), median_data=float(s.data.median()), median_twin=float(s.twin.median()),
                       p=float(wilcoxon(s.data, s.twin).pvalue), frac_bistable=float((reg == "p_bistable").mean()))
    d = P[(P.kind == "data") & P.ch.isin(TW + ["P3", "P4"])].groupby(["sub", "task"])[["th_mu_q50", "th_c_q50", "log_t0_q50", "delta"]].mean().unstack().dropna()
    r["EO-EC dz (posterior mean)"] = {k: float(((d[(k, "restEO")] - d[(k, "restEC")]).mean()) / (d[(k, "restEO")] - d[(k, "restEC")]).std())
                                      for k in ("th_mu_q50", "th_c_q50", "log_t0_q50", "delta")}
    if var:
        T = json.load(open(f"results/topo_tdbrain{sfx}.json"))["all"]
        r["EO-EC map correlation with recorded reference"] = {k: float(pearsonr(list(T[f"EO-EC|{k}"]["dz"].values()), list(topo0[f"EO-EC|{k}"]["dz"].values()))[0])
                                                              for k in ("delta", "th_mu_q50", "th_c_q50", "log_t0_q50")}
        r["EO-EC delta clusters"] = [dict(sign=c["sign"], n_channels=len(c["channels"]), p=c["p"]) for c in T[f"EO-EC|delta"]["clusters"] if c["p"] < 0.05]
        r["EO-EC delta clusters, SNR-adjusted"] = [dict(sign=c["sign"], n_channels=len(c["channels"]), p=c["p"]) for c in T["EO-EC|delta, SNR-adjusted"]["clusters"] if c["p"] < 0.05]
    out[var or "recorded reference"] = r
try:
    A0, A1 = json.load(open("results/anphy_summary.json"))["all"], json.load(open("results/anphy_csd_summary.json"))["all"]
    out["ANPHY"] = {lab: {st: dict(data=A[f"data_twin_chunks|scalp mean"][st]["delta"], twin=A[f"twin|scalp mean"][st]["delta"],
                                   p_bistable=A[f"data_twin_chunks|scalp mean"][st]["p_bistable"]) for st in ("W", "N1", "N2", "N3", "REM")}
                    for lab, A in (("average reference", A0), ("surface Laplacian", A1))}
    for lab, A in (("average reference", A0), ("surface Laplacian", A1)):
        out["ANPHY"][lab]["clusters"] = {k: [dict(sign=c["sign"], n_channels=len(c["channels"]), p=c["p"]) for c in A[k]["clusters"] if c["p"] < 0.05]
                                         for k in ("N2-W|delta", "N2-W|th_c_q50", "N2 data-twin|delta")}
except FileNotFoundError:
    pass
json.dump(out, open("results/variants_summary.json", "w"), indent=1)
print(json.dumps(out, indent=1))

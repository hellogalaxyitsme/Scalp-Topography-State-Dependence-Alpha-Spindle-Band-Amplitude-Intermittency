"""Covariates of the state-axis contrasts in TDBRAIN: (1) availability of vigilance-related covariates (hours slept,
coffee, cigarettes, drugs, session time) and depression severity (BDI); (2) projection of each participant's
age/sex-normalised displacement onto the state axis, compared between MDD and healthy with and without these
covariates; (3) dose-response with BDI within MDD."""
import json, numpy as np, pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr
from ampint.io_tdbrain import load_meta

A = json.load(open("results/state_axis.json"))["axis"]
u = np.array([A["u_mu"], A["u_c"]])
R = pd.read_csv("results/nf_tdbrain_reclevel.csv")
ec = R[R.task == "restEC"].dropna(subset=["age", "gender", "th_mu_q50", "th_c_q50"]).copy()
a = (ec.age - 45) / 15
D = np.column_stack([np.ones(len(ec)), a, a * a, ec.gender])
ref = (ec.group == "HEALTHY").values
for k in ("th_mu_q50", "th_c_q50"):
    b = np.linalg.lstsq(D[ref], ec[k].values[ref], rcond=None)[0]
    ec[k + "_res"] = ec[k].values - D @ b
ec["along"] = ec.th_mu_q50_res * u[0] + ec.th_c_q50_res * u[1]
ec["orth"] = -ec.th_mu_q50_res * u[1] + ec.th_c_q50_res * u[0]
meta = load_meta()
m1 = meta[meta.sessID == 1].drop_duplicates("sub")
COV = ["sleep", "coffee", "cigarette", "drugs", "sessTime", "BDI_pre", "education", "well"]
ec = ec.merge(m1[["sub"] + COV], on="sub", how="left")
out = {"availability": {g: {c: int(ec[ec.group == g][c].notna().sum()) for c in COV} | {"n": int((ec.group == g).sum())}
                        for g in ("HEALTHY", "MDD", "SMC")}}
out["examples"] = {c: ec[c].dropna().astype(str).value_counts().head(6).to_dict() for c in COV}


def hour(s):
    try:
        h, m = str(s).split(":")[:2]; return int(h) + int(m) / 60
    except Exception:
        return np.nan


ec["hour"] = ec.sessTime.map(hour)
for c in ("sleep", "coffee", "cigarette", "BDI_pre", "education", "well"):
    ec[c] = pd.to_numeric(ec[c], errors="coerce").replace(99, np.nan)  # 99 = not reported
d = ec[ec.group.isin(["HEALTHY", "MDD", "SMC"])].copy()
d["mdd"] = (d.group == "MDD").astype(float); d["smc"] = (d.group == "SMC").astype(float)
out["numeric availability"] = {c: int(d[c].notna().sum()) for c in ("sleep", "coffee", "cigarette", "hour", "BDI_pre", "education", "well")}
usable = [c for c in ("sleep", "coffee", "well") if out["numeric availability"][c] > 200]
models = [("no covariates", "along ~ mdd + smc")]
if usable:
    models.append(("+ " + ", ".join(usable), "along ~ mdd + smc + " + " + ".join(usable)))
for name, f in models:
    for y in ("along", "orth"):
        m = smf.ols(f.replace("along", y, 1), data=d).fit()
        out[f"{y} | {name}"] = dict(n=int(m.nobs), mdd=float(m.params["mdd"]), p_mdd=float(m.pvalues["mdd"]),
                                    smc=float(m.params["smc"]), p_smc=float(m.pvalues["smc"]),
                                    **{f"{c}": float(m.params[c]) for c in usable if c in m.params},
                                    **{f"p_{c}": float(m.pvalues[c]) for c in usable if c in m.params})
mdd = d[(d.group == "MDD")].dropna(subset=["BDI_pre"])
out["MDD: spearman(along, BDI_pre)"] = dict(n=int(len(mdd)), rho=float(spearmanr(mdd.along, mdd.BDI_pre)[0]),
                                            p=float(spearmanr(mdd.along, mdd.BDI_pre)[1]))
h = d[d.group == "HEALTHY"].dropna(subset=["sleep"])
out["HEALTHY: spearman(along, hours slept)"] = dict(n=int(len(h)), rho=float(spearmanr(h.along, h.sleep)[0]), p=float(spearmanr(h.along, h.sleep)[1]))
json.dump(out, open("results/axis_covariates.json", "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in out.items() if k not in ("availability", "examples")}, indent=1, default=str))

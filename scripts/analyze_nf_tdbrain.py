"""TDBRAIN V3.1: posterior normal-form coordinates of the alpha rhythm.
input: results/post_tdbrain.csv (nf_apply output on results/nf_tdbrain/*.csv)"""
import sys, json, glob, numpy as np, pandas as pd
from ampint.io_tdbrain import load_meta
from ampint.stats import normative_z, group_effects, fdr, GROUPS

POST_CH = ["O1", "Oz", "O2", "Pz", "P3", "P4"]
NF = ["th_mu_q50", "th_c_q50", "log_t0_q50", "p_focus", "p_bistable", "p_oscillatory", "slow_excess", "slow_flag", "cv"]
SPEC_NF = ["peak", "peak_height", "rho"]


UNFLAGGED_ONLY = "--unflagged" in sys.argv
COVERED_ONLY = "--covered" in sys.argv
SFX = "_unflagged" if UNFLAGGED_ONLY else ("_covered" if COVERED_ONLY else "")


def rec_level(post):
    d = post[(post.kind == "data") & post.ch.isin(POST_CH)]
    if UNFLAGGED_ONLY:  # sensitivity: drop channels where the adequacy test detects a hidden slow variable
        d = d[d.slow_flag == 0]
    if COVERED_ONLY:    # sensitivity: keep channels whose summaries lie within the simulated support (coverage_rows.py)
        d = d[d.cov_pct < 95]
    return d.groupby(["sub", "ses", "task"])[NF + SPEC_NF].mean().reset_index()


def spectral_cov():
    fs = glob.glob("results/chaos_tdbrain/*.csv")
    d = pd.concat([pd.read_csv(f) for f in fs if open(f).readline().strip()], ignore_index=True)
    cols = ["ap_exp", "ap_offset", "rel_delta", "rel_theta", "rel_alpha", "rel_beta"]
    return d.groupby(["sub", "ses", "task"])[cols].mean().reset_index(), cols


def icc_31(a, b):
    """ICC(3,1) consistency for two sessions."""
    X = np.column_stack([a, b]); n = len(X)
    ms_r = 2 * np.var(X.mean(1), ddof=1); ms_e = np.sum((X - X.mean(1, keepdims=True) - X.mean(0) + X.mean()) ** 2) / (n - 1)
    return (ms_r - ms_e) / (ms_r + ms_e)


def main():
    post = pd.read_csv("results/post_tdbrain.csv")
    if COVERED_ONLY:
        post["cov_pct"] = pd.read_csv("results/coverage_tdbrain.csv").cov_pct.values
    out = {}
    # --- surrogate null on real spectra
    sur = post[post.kind.str.startswith("iaaft") & post.ch.isin(POST_CH)]
    dat = post[(post.kind == "data") & post.ch.isin(POST_CH)]
    for name, d in [("data", dat), ("iaaft", sur)]:
        out[f"{name}_summary"] = {t: {k: float(d[d.task == t][k].median()) for k in NF} for t in ["restEC", "restEO"]}
        out[f"{name}_regime_frac"] = {t: d[d.task == t][["p_focus", "p_bistable", "p_oscillatory"]].idxmax(1).value_counts(normalize=True).round(3).to_dict()
                                      for t in ["restEC", "restEO"]}
    rec = rec_level(post)
    spec, scols = spectral_cov()
    rec = rec.merge(spec, on=["sub", "ses", "task"], how="left")
    meta = load_meta()
    m1 = meta[meta.sessID == 1].drop_duplicates("sub")[["sub", "indication", "age", "gender"]]
    df = rec[rec.ses == "ses-1"].merge(m1, on="sub", how="left")
    df = df[(df.age >= 18) & df.gender.notna()].copy()
    df["group"] = df["indication"].where(df["indication"].isin(GROUPS + ["HEALTHY"]))
    # --- healthy EO vs EC within subject
    h = df[df.group == "HEALTHY"].pivot_table(index="sub", columns="task", values=NF + SPEC_NF).dropna()
    out["healthy_EO_minus_EC"] = {k: dict(n=int(len(h)), median_EC=float(h[(k, "restEC")].median()), median_EO=float(h[(k, "restEO")].median()),
                                          dz=float((h[(k, "restEO")] - h[(k, "restEC")]).mean() / (h[(k, "restEO")] - h[(k, "restEC")]).std()))
                                  for k in NF + SPEC_NF}
    allp = df.pivot_table(index="sub", columns="task", values=NF).dropna()
    out["all_EO_minus_EC_dz"] = {k: float((allp[(k, "restEO")] - allp[(k, "restEC")]).mean() / (allp[(k, "restEO")] - allp[(k, "restEC")]).std()) for k in NF}
    # --- clinical normative effects
    res = []
    for task in ["restEC", "restEO"]:
        t = df[df.task == task].copy(); ref = (t.group == "HEALTHY").values
        for k in ["th_mu_q50", "th_c_q50", "log_t0_q50", "p_bistable", "slow_excess"]:
            kinds = [("raw", ()), ("spectrum_adjusted", tuple(SPEC_NF + scols))]
            if k in ("th_mu_q50", "th_c_q50", "log_t0_q50"):
                kinds += [("spectrum_slow_adjusted", tuple(SPEC_NF + scols + ["slow_excess"]))]
            for kind, extra in kinds:
                t["z"] = normative_z(t, k, ref, extra)
                ge = group_effects(t, "z", n_perm=5000, n_boot=2000)
                ge["task"] = task; ge["measure"] = k; ge["kind"] = kind
                res.append(ge)
    res = pd.concat(res, ignore_index=True)
    for kind in res.kind.unique():
        m = res.kind == kind; res.loc[m, "q"] = fdr(res.loc[m, "p"].values)
    res.to_csv(f"results/nf_group_effects{SFX}.csv", index=False)
    out["n_by_group"] = df[df.task == "restEC"].group.value_counts().to_dict()
    # --- reliability (subjects with ses-1 and ses-2)
    RT = NF[:3] + ["p_bistable", "cv", "peak"]
    r12 = rec[rec.ses.isin(["ses-1", "ses-2"])].pivot_table(index=["sub", "task"], columns="ses", values=RT)
    out["icc_ses1_ses2"], out["n_retest"] = {}, {}
    for t in ["restEC", "restEO"]:
        sub_t = r12.xs(t, level="task") if t in r12.index.get_level_values("task") else None
        if sub_t is None:
            continue
        for k in RT:
            pr = sub_t[[(k, "ses-1"), (k, "ses-2")]].dropna()
            if len(pr) > 10:
                out["icc_ses1_ses2"][f"{k}|{t}"] = float(icc_31(pr.iloc[:, 0].values, pr.iloc[:, 1].values))
                out["n_retest"][t] = int(len(pr))
    # --- topography (healthy, per channel medians)
    topo = post[(post.kind == "data")].merge(m1, on="sub", how="left")
    topo = topo[(topo.indication == "HEALTHY") & (topo.ses == "ses-1")]
    topo.groupby(["task", "ch"])[NF + SPEC_NF].median().reset_index().to_csv("results/nf_topography_healthy.csv", index=False)
    df.to_csv(f"results/nf_tdbrain_reclevel{SFX}.csv", index=False)
    json.dump(out, open(f"results/nf_tdbrain_summary{SFX}.json", "w"), indent=1, default=str)
    print(json.dumps(out, indent=1, default=str)[:6000])
    pd.set_option("display.width", 220)
    print(res.sort_values("p").head(30).round(4).to_string())


if __name__ == "__main__":
    main()


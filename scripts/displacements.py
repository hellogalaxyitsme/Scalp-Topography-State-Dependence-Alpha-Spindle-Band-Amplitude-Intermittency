"""Displacement vectors (d mu, d c) in raw unfolding units for physiological, developmental and clinical contrasts,
with bootstrap intervals for each component and for the cosine with the eyes-open direction."""
import json, numpy as np, pandas as pd
rng = np.random.default_rng(0)
NB = 2000
POST_M = ["O1", "O2", "P3", "P4", "Pz"]
POST_D = ["O1", "OZ", "O2", "PZ", "P3", "P4", "POZ", "PO3", "PO4"]
EO = "eyes open - closed (healthy)"


def boot_mean(D):
    D = np.asarray(D, float); D = D[np.isfinite(D).all(1)]
    m = D.mean(0); b = np.array([D[rng.integers(0, len(D), len(D))].mean(0) for _ in range(NB)])
    return dict(d_mu=float(m[0]), d_c=float(m[1]), mu_lo=float(np.percentile(b[:, 0], 2.5)), mu_hi=float(np.percentile(b[:, 0], 97.5)),
                c_lo=float(np.percentile(b[:, 1], 2.5)), c_hi=float(np.percentile(b[:, 1], 97.5)), n=int(len(D))), b


def boot_diff(A, B):
    A = np.asarray(A, float); B = np.asarray(B, float); A = A[np.isfinite(A).all(1)]; B = B[np.isfinite(B).all(1)]
    m = A.mean(0) - B.mean(0)
    b = np.array([A[rng.integers(0, len(A), len(A))].mean(0) - B[rng.integers(0, len(B), len(B))].mean(0) for _ in range(NB)])
    return dict(d_mu=float(m[0]), d_c=float(m[1]), mu_lo=float(np.percentile(b[:, 0], 2.5)), mu_hi=float(np.percentile(b[:, 0], 97.5)),
                c_lo=float(np.percentile(b[:, 1], 2.5)), c_hi=float(np.percentile(b[:, 1], 97.5)), n=f"{len(A)}/{len(B)}"), b


def contrasts():
    """All contrasts: dict of point estimates/intervals and dict of bootstrap replicates (NB x 2)."""
    out, BOOT = {}, {}

    def put(key, res):
        out[key], BOOT[key] = res

    R = pd.read_csv("results/nf_tdbrain_reclevel.csv")
    H = R[R.group == "HEALTHY"]
    w = H.pivot_table(index="sub", columns="task", values=["th_mu_q50", "th_c_q50"]).dropna()
    put(EO, boot_mean(np.column_stack([w[("th_mu_q50", "restEO")] - w[("th_mu_q50", "restEC")],
                                       w[("th_c_q50", "restEO")] - w[("th_c_q50", "restEC")]])))
    # aging: slope x 20 y in healthy EC
    he = H[H.task == "restEC"].dropna(subset=["age", "th_mu_q50", "th_c_q50"])

    def age_slope(d):
        X = np.column_stack([np.ones(len(d)), d.age, d.gender])
        return [np.linalg.lstsq(X, d[k].values, rcond=None)[0][1] * 20 for k in ["th_mu_q50", "th_c_q50"]]
    s0 = age_slope(he); bs = np.array([age_slope(he.iloc[rng.integers(0, len(he), len(he))]) for _ in range(NB)])
    put("aging, +20 y (healthy)", (dict(d_mu=s0[0], d_c=s0[1], mu_lo=float(np.percentile(bs[:, 0], 2.5)), mu_hi=float(np.percentile(bs[:, 0], 97.5)),
                                        c_lo=float(np.percentile(bs[:, 1], 2.5)), c_hi=float(np.percentile(bs[:, 1], 97.5)), n=int(len(he))), bs))
    # clinical TDBRAIN (EC), age/sex-normalised residuals in raw units
    ec = R[R.task == "restEC"].dropna(subset=["age", "gender", "th_mu_q50", "th_c_q50"]).copy()

    def design(d):
        a = (d.age.values - 45) / 15
        return np.column_stack([np.ones(len(d)), a, a * a, d.gender.values])
    ref = ec.group == "HEALTHY"
    for k in ["th_mu_q50", "th_c_q50"]:
        b = np.linalg.lstsq(design(ec[ref]), ec.loc[ref, k].values, rcond=None)[0]
        ec[k + "_res"] = ec[k].values - design(ec) @ b
    for g in ["MDD", "ADHD", "SMC", "OCD", "INSOMNIA", "TINNITUS", "PARKINSON"]:
        put(f"TDBRAIN {g} - healthy", boot_diff(ec[ec.group == g][["th_mu_q50_res", "th_c_q50_res"]].values,
                                                ec[ref][["th_mu_q50_res", "th_c_q50_res"]].values))
    # Mumtaz (EC)
    pm = pd.read_csv("results/post_mumtaz.csv"); pm = pm[pm.ch.isin(POST_M) & (pm.task == "restEC")]
    pm = pm.groupby(["group", "sub"])[["th_mu_q50", "th_c_q50"]].mean().reset_index()
    put("Mumtaz MDD - healthy", boot_diff(pm[pm.group == "MDD"][["th_mu_q50", "th_c_q50"]].values, pm[pm.group == "HEALTHY"][["th_mu_q50", "th_c_q50"]].values))
    # ds003478 (EC)
    pdd = pd.read_csv("results/post_ds003478.csv"); pdd = pdd[pdd.ch.isin(POST_D) & (pdd.task == "restEC")]
    pdd = pdd.groupby(["sub"])[["th_mu_q50", "th_c_q50", "BDI"]].mean().reset_index()
    put("ds003478 high - low BDI", boot_diff(pdd[pdd.BDI >= 13][["th_mu_q50", "th_c_q50"]].values, pdd[pdd.BDI <= 6][["th_mu_q50", "th_c_q50"]].values))
    # sleep (sigma, Fpz-Cz): N2 - W paired
    sg = pd.read_csv("results/sleep_sigma_subject_stage.csv"); sg = sg[sg.ch == "EEG Fpz-Cz"]
    ws = sg.pivot_table(index="subj", columns="stage", values=["th_mu_q50", "th_c_q50"])
    put("sleep spindles N2 - wake (sigma)", boot_mean(np.column_stack([ws[("th_mu_q50", "N2")] - ws[("th_mu_q50", "W")],
                                                                       ws[("th_c_q50", "N2")] - ws[("th_c_q50", "W")]])))
    return out, BOOT


def cos(a, b):
    return np.sum(a * b, -1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12)


if __name__ == "__main__":
    out, BOOT = contrasts()
    # cosine with the eyes-open direction: point estimate and percentile interval over independent bootstrap replicates
    eo, beo = np.array([out[EO]["d_mu"], out[EO]["d_c"]]), BOOT[EO]
    for k, v in out.items():
        if k == EO:
            continue
        cb = cos(BOOT[k], beo)
        v.update(cos_eo=float(cos(np.array([v["d_mu"], v["d_c"]]), eo)), cos_lo=float(np.percentile(cb, 2.5)),
                 cos_hi=float(np.percentile(cb, 97.5)), frac_abs_cos_gt_0_9=float(np.mean(np.abs(cb) > 0.9)))
    json.dump(out, open("results/displacements.json", "w"), indent=1)
    for k, v in out.items():
        print(f"{k:38s} dmu={v['d_mu']:+.2f} [{v['mu_lo']:+.2f},{v['mu_hi']:+.2f}]  dc={v['d_c']:+.2f} [{v['c_lo']:+.2f},{v['c_hi']:+.2f}]  n={v['n']}"
              + (f"  cos={v['cos_eo']:+.3f} [{v['cos_lo']:+.2f},{v['cos_hi']:+.2f}]" if "cos_eo" in v else ""))

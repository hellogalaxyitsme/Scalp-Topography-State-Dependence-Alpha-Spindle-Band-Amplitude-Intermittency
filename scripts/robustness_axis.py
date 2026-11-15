"""Robustness of the clinical state-axis decomposition. Each clinical contrast is recomputed (eyes closed) as
(a) in state_axis.py (age/sex-normalised for TDBRAIN), (b) with spectral covariates in the normalisation,
(c) within the simulated support only, (d) precision-weighted by the posterior interval widths, and
(e) TDBRAIN only, with hours slept, coffee and self-rated well-being. The axis and its bootstrap replicates are
those of state_axis.py (eyes-open displacement); components are tested by bootstrap with FDR within variant."""
import json, numpy as np, pandas as pd
from displacements import contrasts, EO, NB
from ampint.io_tdbrain import load_meta
from ampint.stats import fdr

POST_T = ["O1", "Oz", "O2", "Pz", "P3", "P4"]
POST_M = ["O1", "O2", "P3", "P4", "Pz"]
POST_D = ["O1", "OZ", "O2", "PZ", "P3", "P4", "POZ", "PO3", "PO4"]
SPEC = ["rho", "peak", "peak_height"]
SPEC_T = SPEC + ["ap_exp", "ap_offset", "rel_delta", "rel_theta", "rel_alpha", "rel_beta"]
COLS = ["sub", "ch", "task", "kind", "th_mu_q50", "th_c_q50", "th_mu_lo90", "th_mu_hi90", "th_c_lo90", "th_c_hi90"] + SPEC
rng = np.random.default_rng(1)


def rec_table(post, cov, chans, keys):
    p = post.copy(); p["covered"] = cov
    p = p[p.kind.isin(["data", "nf"]) & p.ch.isin(chans) & (p.task == "restEC")]   # external tables label rows 'nf'
    p["var"] = ((p.th_mu_hi90 - p.th_mu_lo90) / 3.29) ** 2 + ((p.th_c_hi90 - p.th_c_lo90) / 3.29) ** 2
    agg = lambda d: d.groupby(keys)[["th_mu_q50", "th_c_q50", "var"] + SPEC].mean().reset_index()
    return agg(p), agg(p[p.covered])


def residualise(df, ref, extra):
    """Residuals of mu and c from a reference-group linear model (intercept + extra covariates, standardised)."""
    X = [np.ones(len(df))] + [((df[e] - df[e].mean()) / (df[e].std() + 1e-12)).values for e in extra]
    X = np.column_stack(X); out = {}
    ok = np.isfinite(X).all(1) & df[["th_mu_q50", "th_c_q50"]].notna().all(1).values
    for k in ("th_mu_q50", "th_c_q50"):
        b = np.linalg.lstsq(X[ok & ref], df[k].values[ok & ref], rcond=None)[0]
        out[k] = np.where(ok, df[k].values - X @ b, np.nan)
    return np.column_stack([out["th_mu_q50"], out["th_c_q50"]]), ok


def diff_boot(Y, wgt, a, b):
    """Weighted mean difference of rows a minus rows b with recording bootstrap (NB replicates)."""
    A, B, wa, wb = Y[a], Y[b], wgt[a], wgt[b]
    m = lambda Z, w: (Z * w[:, None]).sum(0) / w.sum()
    est = m(A, wa) - m(B, wb); bs = np.empty((NB, 2))
    for i in range(NB):
        ia, ib = rng.integers(0, len(A), len(A)), rng.integers(0, len(B), len(B))
        bs[i] = m(A[ia], wa[ia]) - m(B[ib], wb[ib])
    return est, bs


def decompose(est, bs, u, ub):
    perp = lambda v: np.stack([-v[..., 1], v[..., 0]], -1)
    al, orb = np.sum(bs * ub, 1), np.sum(bs * perp(ub), 1)
    p = lambda v: float(min(1.0, max(1.0 / len(v), 2 * min(np.mean(v <= 0), np.mean(v >= 0)))))
    return dict(d_mu=float(est[0]), d_c=float(est[1]), along=float(est @ u), along_lo=float(np.percentile(al, 2.5)),
                along_hi=float(np.percentile(al, 97.5)), p_along=p(al), orth=float(est @ perp(u)),
                orth_lo=float(np.percentile(orb, 2.5)), orth_hi=float(np.percentile(orb, 97.5)), p_orth=p(orb))


def main():
    out, BOOT = contrasts()
    u = np.array([out[EO]["d_mu"], out[EO]["d_c"]]); u /= np.linalg.norm(u)
    ub = BOOT[EO] / np.linalg.norm(BOOT[EO], axis=1, keepdims=True)
    res = {}
    # TDBRAIN
    P = pd.read_csv("results/post_tdbrain.csv", usecols=COLS + ["ses"])
    cov = pd.read_csv("results/coverage_tdbrain.csv").cov_pct.values < 95
    keep = (P.ses == "ses-1").values
    T, Tc = rec_table(P[keep], cov[keep], POST_T, ["sub"])
    R = pd.read_csv("results/nf_tdbrain_reclevel.csv")
    R = R[R.task == "restEC"][["sub", "group", "age", "gender"] + SPEC_T[3:]]
    meta = load_meta(); m1 = meta[meta.sessID == 1].drop_duplicates("sub")[["sub", "sleep", "coffee", "well"]]
    for c in ("sleep", "coffee", "well"):
        m1[c] = pd.to_numeric(m1[c], errors="coerce").replace(99, np.nan)
    for name, tab in (("TDBRAIN", T), ("TDBRAIN covered", Tc)):
        d = tab.merge(R, on="sub").merge(m1, on="sub", how="left").dropna(subset=["group", "age", "gender"])
        d["a"], d["a2"] = (d.age - 45) / 15, ((d.age - 45) / 15) ** 2
        ref = (d.group == "HEALTHY").values
        variants = {"age/sex": ["a", "a2", "gender"]} if "covered" in name else {
            "age/sex": ["a", "a2", "gender"], "+ spectral covariates": ["a", "a2", "gender"] + SPEC_T,
            "+ sleep, coffee, well-being": ["a", "a2", "gender", "sleep", "coffee", "well"]}
        for vname, ex in variants.items():
            Y, ok = residualise(d, ref, ex)
            for weighted in ((False, True) if vname == "age/sex" and "covered" not in name else (False,)):
                w = 1 / d["var"].values if weighted else np.ones(len(d))
                vtag = f"{name} | {vname}" + (" | precision-weighted" if weighted else "")
                for g in ["MDD", "ADHD", "SMC", "OCD", "INSOMNIA", "TINNITUS", "PARKINSON"]:
                    a, b = ok & (d.group == g).values, ok & ref
                    if a.sum() < 10:
                        continue
                    est, bs = diff_boot(Y, w, a, b)
                    res.setdefault(vtag, {})[g] = dict(n=f"{a.sum()}/{b.sum()}", **decompose(est, bs, u, ub))
    # external cohorts (eyes closed): controls as reference; no age/sex normalisation, as in displacements.py
    for name, f, cf, chans, grp in (("Mumtaz", "results/post_mumtaz.csv", "results/coverage_mumtaz.csv", POST_M, "group"),
                                    ("ds003478", "results/post_ds003478.csv", "results/coverage_ds003478.csv", POST_D, "BDI")):
        P = pd.read_csv(f, usecols=COLS + [grp] if grp == "BDI" else COLS + ["group"])
        cv = pd.read_csv(cf).cov_pct.values < 95
        keys = ["sub", grp] if grp == "BDI" else ["group", "sub"]
        tab, tabc = rec_table(P, cv, chans, keys)
        for tname, d in (("", tab), (" covered", tabc)):
            if grp == "BDI":
                a_, b_ = (d.BDI >= 13).values, (d.BDI <= 6).values
            else:
                a_, b_ = (d.group == "MDD").values, (d.group == "HEALTHY").values
            for vname, ex in (("raw", []), ("+ spectral covariates", SPEC)):
                if tname and ex:
                    continue
                Y, ok = residualise(d, b_, ex)
                for weighted in ((False, True) if not tname and not ex else (False,)):
                    w = 1 / d["var"].values if weighted else np.ones(len(d))
                    est, bs = diff_boot(Y, w, ok & a_, ok & b_)
                    vtag = f"{name}{tname} | {vname}" + (" | precision-weighted" if weighted else "")
                    res.setdefault(vtag, {})["depression"] = dict(n=f"{(ok & a_).sum()}/{(ok & b_).sum()}", **decompose(est, bs, u, ub))
    for vtag, gs in res.items():
        if len(gs) > 1:
            for kind in ("along", "orth"):
                q = fdr([v[f"p_{kind}"] for v in gs.values()])
                for v, qq in zip(gs.values(), q):
                    v[f"q_{kind}"] = float(qq)
    json.dump(res, open("results/robustness_axis.json", "w"), indent=1)
    for vtag, gs in res.items():
        for g, v in gs.items():
            if g in ("MDD", "SMC", "OCD", "PARKINSON", "depression"):
                print(f"{vtag:58s} {g:10s} n={v['n']:>8}  along={v['along']:+.2f} [{v['along_lo']:+.2f},{v['along_hi']:+.2f}] p={v['p_along']:.4f}"
                      f"  orth={v['orth']:+.2f} [{v['orth_lo']:+.2f},{v['orth_hi']:+.2f}] p={v['p_orth']:.4f}")


if __name__ == "__main__":
    main()

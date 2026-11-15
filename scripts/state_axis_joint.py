"""State-axis decomposition with a joint bootstrap. The healthy TDBRAIN participants define both the eyes-open
axis and the reference of every TDBRAIN clinical contrast, so in each bootstrap replicate the healthy participants
are resampled once (each carrying its eyes-closed and eyes-open recordings) and the same draw is used for the axis,
for the reference model (age/sex or covariate adjustment, refitted in every replicate) and for the healthy side of
the contrast. Clinical groups are resampled separately. Independent datasets (Mumtaz, ds003478, Sleep-EDF,
ANPHY-Sleep) are resampled independently and projected onto the axis replicate of the same index.
Main contrasts as in state_axis.py, robustness variants as in robustness_axis.py.
usage: python scripts/state_axis_joint.py  -> results/state_axis_joint.json, results/robustness_axis_joint.json"""
import json, numpy as np, pandas as pd
from displacements import cos, EO, NB, POST_M, POST_D
from robustness_axis import rec_table, POST_T, SPEC, SPEC_T, COLS
from ampint.io_tdbrain import load_meta
from ampint.stats import fdr

rng = np.random.default_rng(11)
GROUPS = ["MDD", "ADHD", "SMC", "OCD", "INSOMNIA", "TINNITUS", "PARKINSON"]
ANPHY_POST = ["O1", "OZ", "O2", "POZ", "PZ", "PO3", "PO4"]
MC = ["th_mu_q50", "th_c_q50"]
perp = lambda v: np.stack([-v[..., 1], v[..., 0]], -1)


def pboot(v):
    return float(min(1.0, max(1.0 / len(v), 2 * min(np.mean(v <= 0), np.mean(v >= 0)))))


def wmean(Z, w):
    return (Z * w[:, None]).sum(0) / w.sum()


def adjusted_contrast(d, ref, grp, extra, w, ref_draws=None):
    """Weighted mean residual of grp minus ref, residuals from a model (intercept + extra) fitted to ref.
    The model is refitted in every bootstrap replicate. ref_draws: list of NB row-index arrays for the reference
    (joint healthy draws); otherwise the reference is resampled independently."""
    X = np.column_stack([np.ones(len(d))] + [((d[e] - d[e].mean()) / (d[e].std() + 1e-12)).values for e in extra])
    Y = d[MC].values.astype(float)
    ok = np.isfinite(X).all(1) & np.isfinite(Y).all(1)
    ri, gi = np.where(ok & ref)[0], np.where(ok & grp)[0]

    def est(r, g):
        beta = np.linalg.lstsq(X[r], Y[r], rcond=None)[0]
        return wmean(Y[g] - X[g] @ beta, w[g]) - wmean(Y[r] - X[r] @ beta, w[r])
    bs = np.empty((NB, 2))
    for b in range(NB):
        r = ref_draws[b][ok[ref_draws[b]] & ref[ref_draws[b]]] if ref_draws is not None else rng.choice(ri, len(ri))
        bs[b] = est(r, rng.choice(gi, len(gi)))
    return est(ri, gi), bs, f"{len(gi)}/{len(ri)}"


def paired_boot(D):
    D = np.asarray(D, float); D = D[np.isfinite(D).all(1)]
    return D.mean(0), np.array([D[rng.integers(0, len(D), len(D))].mean(0) for _ in range(NB)]), int(len(D))


def decompose(n, est, bs, u, ub, with_cos=True):
    al, orb = np.sum(bs * ub, 1), np.sum(bs * perp(ub), 1)
    r = dict(n=n, d_mu=float(est[0]), d_c=float(est[1]), along=float(est @ u), along_lo=float(np.percentile(al, 2.5)),
             along_hi=float(np.percentile(al, 97.5)), p_along=pboot(al), orth=float(est @ perp(u)),
             orth_lo=float(np.percentile(orb, 2.5)), orth_hi=float(np.percentile(orb, 97.5)), p_orth=pboot(orb))
    if with_cos:
        cb = cos(bs, ub)
        r.update(cos_eo=float(cos(est, u)), cos_lo=float(np.percentile(cb, 2.5)), cos_hi=float(np.percentile(cb, 97.5)))
    return r


def draws_for(table_subs, hs, hdraw):
    """Map the healthy draws (indices into hs) to row indices of a table keyed by subject."""
    row = {s: i for i, s in enumerate(table_subs)}
    return [np.array([row[hs[k]] for k in idx if hs[k] in row], int) for idx in hdraw]


def main():
    R = pd.read_csv("results/nf_tdbrain_reclevel.csv")
    ec = R[R.task == "restEC"].dropna(subset=["age", "gender"] + MC).reset_index(drop=True)
    ec["a"] = (ec.age - 45) / 15; ec["a2"] = ec.a ** 2; ec["gender"] = ec.gender.astype(float)
    hs = ec[ec.group == "HEALTHY"]["sub"].tolist()
    H = R[R.group == "HEALTHY"].pivot_table(index="sub", columns="task", values=MC).dropna()
    pair = {s: np.array([H.loc[s, ("th_mu_q50", "restEO")] - H.loc[s, ("th_mu_q50", "restEC")],
                         H.loc[s, ("th_c_q50", "restEO")] - H.loc[s, ("th_c_q50", "restEC")]]) for s in H.index}
    hdraw = [rng.integers(0, len(hs), len(hs)) for _ in range(NB)]
    eo_pts = np.array([pair[s] for s in hs if s in pair])
    eo = eo_pts.mean(0)
    beo = np.array([np.mean([pair[hs[k]] for k in idx if hs[k] in pair], 0) for idx in hdraw])
    u = eo / np.linalg.norm(eo); ub = beo / np.linalg.norm(beo, axis=1, keepdims=True)
    res = dict(axis=dict(u_mu=float(u[0]), u_c=float(u[1]), angle_deg=float(np.degrees(np.arctan2(u[1], u[0]))),
                         n_healthy_pool=len(hs), n_axis=int(len(eo_pts)),
                         bootstrap="joint: healthy participants resampled once per replicate for axis and TDBRAIN references"))
    res[EO] = decompose(int(len(eo_pts)), eo, beo, u, ub)
    # aging (+20 y) in healthy eyes closed, same healthy draws
    he = ec[ec.group == "HEALTHY"].reset_index(drop=True)
    sl = lambda d: np.array([np.linalg.lstsq(np.column_stack([np.ones(len(d)), d.age, d.gender]), d[k].values, rcond=None)[0][1] * 20 for k in MC])
    res["aging, +20 y (healthy)"] = decompose(int(len(he)), sl(he), np.array([sl(he.iloc[idx]) for idx in hdraw]), u, ub)
    # TDBRAIN clinical contrasts (age/sex model refitted on the healthy draw)
    ref = (ec.group == "HEALTHY").values; rd = draws_for(ec["sub"].tolist(), hs, hdraw)
    for g in GROUPS:
        est, bs, n = adjusted_contrast(ec, ref, (ec.group == g).values, ["a", "a2", "gender"], np.ones(len(ec)), rd)
        res[f"TDBRAIN {g} - healthy"] = decompose(n, est, bs, u, ub)
    # external cohorts (independent) projected onto the axis replicates
    pm = pd.read_csv("results/post_mumtaz.csv"); pm = pm[pm.ch.isin(POST_M) & (pm.task == "restEC")]
    pm = pm.groupby(["group", "sub"])[MC].mean().reset_index()
    est, bs, n = adjusted_contrast(pm, (pm.group == "HEALTHY").values, (pm.group == "MDD").values, [], np.ones(len(pm)))
    res["Mumtaz MDD - healthy"] = decompose(n, est, bs, u, ub)
    pdd = pd.read_csv("results/post_ds003478.csv"); pdd = pdd[pdd.ch.isin(POST_D) & (pdd.task == "restEC")]
    pdd = pdd.groupby("sub")[MC + ["BDI"]].mean().reset_index()
    est, bs, n = adjusted_contrast(pdd, (pdd.BDI <= 6).values, (pdd.BDI >= 13).values, [], np.ones(len(pdd)))
    res["ds003478 high - low BDI"] = decompose(n, est, bs, u, ub)
    sg = pd.read_csv("results/sleep_sigma_subject_stage.csv"); sg = sg[sg.ch == "EEG Fpz-Cz"]
    ws = sg.pivot_table(index="subj", columns="stage", values=MC)
    est, bs, n = paired_boot(np.column_stack([ws[(k, "N2")] - ws[(k, "W")] for k in MC]))
    res["sleep spindles N2 - wake (sigma)"] = decompose(n, est, bs, u, ub)
    sa = pd.read_csv("results/sleep_alpha_subject_stage.csv")
    for ch, tag in (("EEG Pz-Oz", "Pz-Oz"), ("EEG Fpz-Cz", "Fpz-Cz")):
        w = sa[sa.ch == ch].pivot_table(index="subj", columns="stage", values=MC)
        est, bs, n = paired_boot(np.column_stack([w[(k, "N1")] - w[(k, "W")] for k in MC]))
        res[f"alpha wake -> N1 (Sleep-EDF {tag})"] = decompose(n, est, bs, u, ub)
    aa = pd.read_csv("results/anphy_alpha_subject_stage.csv")
    aa = aa[aa.ch.isin(ANPHY_POST)].groupby(["rec", "stage"])[MC].mean().reset_index()
    w = aa.pivot_table(index="rec", columns="stage", values=MC)
    est, bs, n = paired_boot(np.column_stack([w[(k, "N1")] - w[(k, "W")] for k in MC]))
    res["alpha wake -> N1 (ANPHY posterior)"] = decompose(n, est, bs, u, ub)
    clin = [k for k in res if k.startswith(("TDBRAIN", "Mumtaz", "ds003478"))]
    for kind in ("along", "orth"):
        for k, q in zip(clin, fdr([res[k][f"p_{kind}"] for k in clin])):
            res[k][f"q_{kind}"] = float(q)
    json.dump(res, open("results/state_axis_joint.json", "w"), indent=1)
    for k, r in res.items():
        if k != "axis":
            print(f"{k:40s} n={r['n']!s:>8} along={r['along']:+.2f} [{r['along_lo']:+.2f},{r['along_hi']:+.2f}] p={r['p_along']:.4f}"
                  f"{' q=%.3f' % r['q_along'] if 'q_along' in r else '':8s} orth={r['orth']:+.2f} [{r['orth_lo']:+.2f},{r['orth_hi']:+.2f}] "
                  f"p={r['p_orth']:.4f}{' q=%.3f' % r['q_orth'] if 'q_orth' in r else ''}  cos={r['cos_eo']:+.3f} [{r['cos_lo']:+.2f},{r['cos_hi']:+.2f}]")

    # robustness variants (as robustness_axis.py), joint for TDBRAIN
    rob = {}
    P = pd.read_csv("results/post_tdbrain.csv", usecols=COLS + ["ses"])
    cov = pd.read_csv("results/coverage_tdbrain.csv").cov_pct.values < 95
    keep = (P.ses == "ses-1").values
    T, Tc = rec_table(P[keep], cov[keep], POST_T, ["sub"])
    Rc = R[R.task == "restEC"][["sub", "group", "age", "gender"] + SPEC_T[3:]]
    meta = load_meta(); m1 = meta[meta.sessID == 1].drop_duplicates("sub")[["sub", "sleep", "coffee", "well"]]
    for c in ("sleep", "coffee", "well"):
        m1[c] = pd.to_numeric(m1[c], errors="coerce").replace(99, np.nan)
    for name, tab in (("TDBRAIN", T), ("TDBRAIN covered", Tc)):
        d = tab.merge(Rc, on="sub").merge(m1, on="sub", how="left").dropna(subset=["group", "age", "gender"]).reset_index(drop=True)
        d["a"], d["a2"] = (d.age - 45) / 15, ((d.age - 45) / 15) ** 2
        d["gender"] = d.gender.astype(float)
        ref = (d.group == "HEALTHY").values; rd = draws_for(d["sub"].tolist(), hs, hdraw)
        variants = {"age/sex": ["a", "a2", "gender"]} if "covered" in name else {
            "age/sex": ["a", "a2", "gender"], "+ spectral covariates": ["a", "a2", "gender"] + SPEC_T,
            "+ sleep, coffee, well-being": ["a", "a2", "gender", "sleep", "coffee", "well"]}
        for vname, ex in variants.items():
            for weighted in ((False, True) if vname == "age/sex" and "covered" not in name else (False,)):
                w = 1 / d["var"].values if weighted else np.ones(len(d))
                vtag = f"{name} | {vname}" + (" | precision-weighted" if weighted else "")
                for g in GROUPS:
                    if ((d.group == g).sum()) < 10:
                        continue
                    est, bs, n = adjusted_contrast(d, ref, (d.group == g).values, ex, w, rd)
                    rob.setdefault(vtag, {})[g] = decompose(n, est, bs, u, ub, with_cos=False)
    for name, f, cf, chans, grp in (("Mumtaz", "results/post_mumtaz.csv", "results/coverage_mumtaz.csv", POST_M, "group"),
                                    ("ds003478", "results/post_ds003478.csv", "results/coverage_ds003478.csv", POST_D, "BDI")):
        P = pd.read_csv(f, usecols=COLS + [grp] if grp == "BDI" else COLS + ["group"])
        cv = pd.read_csv(cf).cov_pct.values < 95
        keys = ["sub", grp] if grp == "BDI" else ["group", "sub"]
        tab, tabc = rec_table(P, cv, chans, keys)
        for tname, d in (("", tab), (" covered", tabc)):
            d = d.reset_index(drop=True)
            a_, b_ = ((d.BDI >= 13).values, (d.BDI <= 6).values) if grp == "BDI" else ((d.group == "MDD").values, (d.group == "HEALTHY").values)
            for vname, ex in (("raw", []), ("+ spectral covariates", SPEC)):
                if tname and ex:
                    continue
                for weighted in ((False, True) if not tname and not ex else (False,)):
                    w = 1 / d["var"].values if weighted else np.ones(len(d))
                    est, bs, n = adjusted_contrast(d, b_, a_, ex, w)
                    vtag = f"{name}{tname} | {vname}" + (" | precision-weighted" if weighted else "")
                    rob.setdefault(vtag, {})["depression"] = decompose(n, est, bs, u, ub, with_cos=False)
    for vtag, gs in rob.items():
        if len(gs) > 1:
            for kind in ("along", "orth"):
                for v, q in zip(gs.values(), fdr([v[f"p_{kind}"] for v in gs.values()])):
                    v[f"q_{kind}"] = float(q)
    json.dump(rob, open("results/robustness_axis_joint.json", "w"), indent=1)
    for vtag, gs in rob.items():
        for g, v in gs.items():
            if g in ("MDD", "SMC", "depression"):
                print(f"{vtag:58s} {g:10s} n={v['n']:>8}  along={v['along']:+.2f} [{v['along_lo']:+.2f},{v['along_hi']:+.2f}] p={v['p_along']:.4f}"
                      f"  orth={v['orth']:+.2f} [{v['orth_lo']:+.2f},{v['orth_hi']:+.2f}] p={v['p_orth']:.4f}")


if __name__ == "__main__":
    main()

"""Sensitivity analyses of the state contrasts:
(1) linear twins whose peak frequency and background fraction are re-estimated on the twin (tdbrain_checks.py);
(2) the spindle-band contrasts in a fixed 11.5-14.5 Hz band (sleep_checks.py);
(3) ANPHY-Sleep N2 minus wakefulness for credible peaks, and the range of the SNR changes;
(4) cluster tests on standardized effects: channel-wise thresholds for each channel's own sample size, cluster
    mass as the sum of |d_z|;
(5) state-axis alignment for credible peaks, within the simulated support and at zero SNR change, the bipolar
    ANPHY-Sleep Pz-Oz derivation, and single spectral covariates of the external depression contrasts;
(6) mu_t: effect sizes, reliability and its channel-wise share of the eye-opening change in delta;
(7) N2 minus wakefulness by derivation in both sleep datasets;
(8) DFA exponent of the amplitude envelope as an additional direct statistic;
(9) burst predictions with posterior draws, the burst-duration bias and its recovery in simulations.
usage: python scripts/sensitivity_checks.py [n_perm]  -> results/sensitivity_checks.json"""
import sys, os, glob, json, numpy as np, pandas as pd, mne
from scipy.stats import spearmanr, wilcoxon, mannwhitneyu, t as tdist
from scipy.sparse.csgraph import connected_components
from ampint.nf import fold_delta
from ampint.io_tdbrain import load_meta, EEG26
from ampint.io_anphy import positions
from ampint.stats import channel_adjacency, cluster_test
from descriptor_benchmark import tdbrain_table, icc21, dz, DESC, STATS
from robustness_axis import rec_table, SPEC, COLS, POST_M, POST_D

NPERM = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
NB = 2000
rng = np.random.default_rng(5)
PRIOR_F = {"alpha": (8.0, 12.5), "alpha100": (8.0, 12.0), "sigma": (11.5, 14.5)}
HMIN = 0.1
POST4 = ["O1", "OZ", "O2", "PZ"]
POST6 = ["O1", "OZ", "O2", "PZ", "P3", "P4"]
ANPHY_POST = ["O1", "OZ", "O2", "POZ", "PZ", "PO3", "PO4"]
MC = ["th_mu_q50", "th_c_q50"]
REGIONS = {"frontal": ["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "FC3", "FCZ", "FC4"],
           "central and temporal": ["T7", "C3", "CZ", "C4", "T8", "CP3", "CPZ", "CP4"],
           "parietal and occipital": ["P7", "P3", "PZ", "P4", "P8", "O1", "OZ", "O2"]}


def ci(f, n):
    v = np.array([f(rng.integers(0, n, n)) for _ in range(NB)], float)
    return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]


def dzci(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return dict(n=int(len(x)), dz=dz(x), ci=ci(lambda i: dz(x[i]), len(x)))


def credible(d, band):
    lo, hi = PRIOR_F[band]
    return d.peak.between(lo, hi) & (d.peak_height >= HMIN)


def healthy_adults():
    meta = load_meta(); m1 = meta[meta.sessID == 1].drop_duplicates("sub")
    return set(m1.loc[(m1.indication == "HEALTHY") & (m1.age >= 18), "sub"])


def tdbrain_post(cols, healthy=True):
    P = pd.read_csv("results/post_tdbrain.csv", usecols=list(dict.fromkeys(["sub", "ses", "task", "ch", "kind"] + cols)))
    P["covered"] = pd.read_csv("results/coverage_tdbrain.csv").cov_pct.values < 95
    P["CH"] = P.ch.str.upper()
    if "th_mu_q50" in cols:
        P["delta"] = fold_delta(P.th_mu_q50.values, P.th_c_q50.values)
    if healthy:
        P = P[(P.ses == "ses-1") & P["sub"].isin(healthy_adults())]
    return P


def tdbrain_adj():
    mont = mne.channels.make_standard_montage("standard_1020").get_positions()["ch_pos"]
    pos = {k.upper(): v for k, v in mont.items()}
    chans = [c.upper() for c in EEG26]
    return chans, channel_adjacency(chans, pos)


def anphy_scalp():
    pos = positions()
    S = pd.read_csv("results/post_anphy_sigma.csv", usecols=["ch"]).ch.unique()
    scalp = [c for c in sorted(S) if c in pos]
    return scalp, channel_adjacency(scalp, pos)


def sleep_table(t, cols):
    S = pd.read_csv(f"results/post_{t}.csv", usecols=list(dict.fromkeys(["rec", "ch", "stage", "chunk", "kind"] + cols)))
    if os.path.exists(f"results/coverage_{t}.csv"):
        S["covered"] = pd.read_csv(f"results/coverage_{t}.csv").cov_pct.values < 95
    if "th_mu_q50" in cols:
        S["delta"] = fold_delta(S.th_mu_q50.values, S.th_c_q50.values)
    S["unit"] = S.rec.str[:5] if t.startswith("sleep") else S.rec
    return S


def unit_stage(S, chans, k, agg="median"):
    """Participant x stage value: per participant, channel and stage aggregate over pieces, then mean over channels."""
    g = S[S.ch.isin(chans)].groupby(["unit", "ch", "stage"])[k].agg(agg).reset_index()
    return g.groupby(["unit", "stage"])[k].mean().unstack()


def clusters_list(cl, chans):
    return [dict(sign=c["sign"], n_channels=len(c["channels"]), channels=[chans[i] for i in c["channels"]], p=c["p"])
            for c in cl if c["p"] < 0.05]


def cluster_dz(D, adj, n_perm=NPERM, seed=1, alpha_thr=0.05):
    """Cluster permutation test on standardized paired effects (participants x channels, NaN-tolerant). A channel
    enters a cluster when its paired t exceeds the two-sided threshold for its own number of participants, and the
    cluster mass is the sum of |d_z|; the null distribution uses the same participant sign flips for all channels."""
    r = np.random.default_rng(seed)
    n = np.sum(np.isfinite(D), 0)
    thr = tdist.ppf(1 - alpha_thr / 2, np.maximum(n - 1, 1))

    def found(X):
        e = np.nanmean(X, 0) / np.nanstd(X, 0, ddof=1)
        t = e * np.sqrt(n); out = []
        for sgn in (1, -1):
            idx = np.where(sgn * t > thr)[0]
            if idx.size:
                _, lab = connected_components(adj[np.ix_(idx, idx)], directed=False)
                out += [(sgn, idx[lab == k], float(np.sum(np.abs(e[idx[lab == k]])))) for k in np.unique(lab)]
        return out
    obs = found(D)
    null = np.array([max([m for _, _, m in found(D * r.choice([-1.0, 1.0], size=(D.shape[0], 1)))], default=0.0)
                     for _ in range(n_perm)])
    cl = [dict(sign=int(s), channels=mem.tolist(), mass=m, p=float((1 + np.sum(null >= m)) / (n_perm + 1))) for s, mem, m in obs]
    return sorted(cl, key=lambda c: c["p"])


# (1) twins with re-estimated peak frequency and background fraction
def twins_reestimated():
    f = "results/post_tdbrain_checks.csv"
    if not os.path.exists(f):
        return "missing " + f
    C = pd.read_csv(f, usecols=["sub", "ses", "task", "ch", "kind", "peak", "rho_used", "th_mu_q50", "th_c_q50"])
    C["CH"] = C.ch.str.upper(); C["delta"] = fold_delta(C.th_mu_q50.values, C.th_c_q50.values)
    P = tdbrain_post(["peak", "rho", "th_mu_q50", "th_c_q50"])
    P = P[P.CH.isin(POST4)]
    k = ["sub", "task", "CH"]
    E = P[P.kind == "data"].set_index(k)
    tabs = {"EEG": E, "twin (original)": P[P.kind == "iaaft0"].set_index(k)}
    for kind in ("twin_matched", "twin_own"):
        tabs[kind.replace("_", " ")] = C[(C.kind == kind) & (C.ses == "ses-1")].set_index(k)
    idx = E.index
    for v in tabs.values():
        idx = idx.intersection(v.index)
    out = {"n channel recordings": int(len(idx))}
    shift = tabs["twin own"].loc[idx]
    out["twin own minus EEG, channel level"] = dict(
        median_abs_peak_shift_hz=float((shift.peak - E.loc[idx].peak).abs().median()),
        frac_peak_shift_gt_0_5hz=float(((shift.peak - E.loc[idx].peak).abs() > 0.5).mean()),
        median_background_fraction_change=float((shift.rho_used - E.loc[idx].rho).median()))
    M = {name: v.loc[idx].groupby(level=["sub", "task"])[["delta", "th_c_q50"]].mean() for name, v in tabs.items()}
    for task in ("restEC", "restEO"):
        r = {}
        for name, m in M.items():
            x = m.xs(task, level="task")
            r[name] = dict(n=int(len(x)), median_delta=float(x.delta.median()), median_c=float(x.th_c_q50.median()))
        e, o = M["EEG"].xs(task, level="task").delta, M["twin own"].xs(task, level="task").delta
        r["EEG vs twin own"] = dict(p_wilcoxon=float(wilcoxon(e, o).pvalue), frac_eeg_higher=float((e > o).mean()))
        out[task] = r
    for name in ("twin original", "twin matched", "twin own"):
        tn = name.replace("twin original", "twin (original)")
        d = (M["EEG"].delta - M[tn].delta).unstack("task").dropna()
        out[f"EO-EC of EEG minus {name}"] = dzci((d.restEO - d.restEC).values)
    return out


# (2) fixed 11.5-14.5 Hz band in the sleep data
def fixed_band(scalp, adj):
    out = {}
    for ds, chans in (("anphy", scalp), ("sleep", ["EEG Fpz-Cz"])):
        f = f"results/post_sleep_checks_{ds}.csv"
        if not os.path.exists(f):
            out[ds] = "missing " + f; continue
        S = pd.read_csv(f, usecols=["rec", "ch", "stage", "chunk", "kind", "peak", "th_mu_q50", "th_c_q50"])
        S["delta"] = fold_delta(S.th_mu_q50.values, S.th_c_q50.values)
        S["unit"] = S.rec.str[:5] if ds == "sleep" else S.rec
        D, T = S[S.kind == "data"], S[S.kind == "iaaft0"]
        Dm = D.merge(T[["rec", "ch", "stage", "chunk"]], on=["rec", "ch", "stage", "chunk"])
        G, Gt, Gm = unit_stage(D, chans, "delta"), unit_stage(T, chans, "delta"), unit_stage(Dm, chans, "delta")
        r = {"median delta by stage": {s: dict(eeg=float(G[s].median()), twin=float(Gt[s].median())) for s in ("W", "N2")},
             "N2-W, EEG": dzci((G.N2 - G.W).values), "N2-W, twin": dzci((Gt.N2 - Gt.W).values)}
        for s in ("W", "N2"):
            x = pd.concat([Gm[s], Gt[s]], axis=1, keys=["e", "t"]).dropna()
            r[f"EEG minus twin, {s}"] = dict(n=int(len(x)), median_difference=float((x.e - x.t).median()),
                                            frac_positive=float((x.e > x.t).mean()), p_wilcoxon=float(wilcoxon(x.e, x.t).pvalue))
        if ds == "anphy":
            g = D[D.ch.isin(scalp)].groupby(["rec", "ch", "stage"]).delta.median().unstack()
            Dd = (g.N2 - g.W).unstack("ch").reindex(columns=scalp)
            _, cl = cluster_test(Dd.values, adj, n_perm=NPERM, seed=3)
            r["N2-W clusters"] = clusters_list(cl, scalp)
        out[ds] = r
    return out


# (3) ANPHY-Sleep credible peaks and SNR changes
def credible_and_snr(scalp):
    out = {}
    S = sleep_table("anphy_sigma", ["peak", "peak_height", "th_mu_q50", "th_c_q50", "th_snr_db_hat"])
    S = S[S.kind == "data"]; S["cred"] = credible(S, "sigma")
    for tag, d in (("all pieces", S), ("credible peaks", S[S.cred])):
        G = unit_stage(d, scalp, "delta")
        out[f"ANPHY N2-W delta, scalp mean, {tag}"] = dzci((G.N2 - G.W).values)
    snr = {}
    P = tdbrain_post(["th_snr_db_hat"])
    P = P[P.kind == "data"]
    w = P.pivot_table(index=["sub", "CH"], columns="task", values="th_snr_db_hat").dropna()
    d = (w.restEO - w.restEC)
    per = d.groupby(level="CH").agg(lambda x: float((x > 0).mean()))
    q = d.groupby(level=["sub"]).mean()
    snr["TDBRAIN EO-EC, participant mean over 26 channels"] = dict(
        n=int(len(q)), median=float(q.median()), p5=float(q.quantile(.05)), p95=float(q.quantile(.95)), frac_positive=float((q > 0).mean()))
    snr["TDBRAIN EO-EC, fraction of participants with a positive change, per channel"] = dict(
        min=float(per.min()), max=float(per.max()), median=float(per.median()))
    snr["TDBRAIN EO-EC, channel-recording level"] = dict(p5=float(d.quantile(.05)), median=float(d.median()), p95=float(d.quantile(.95)))
    G = unit_stage(S, scalp, "th_snr_db_hat")
    x = (G.N2 - G.W).dropna()
    snr["ANPHY N2-W, scalp mean"] = dict(n=int(len(x)), median=float(x.median()), p5=float(x.quantile(.05)), p95=float(x.quantile(.95)), frac_positive=float((x > 0).mean()))
    E = sleep_table("sleep_sigma", ["th_snr_db_hat"]); E = E[E.kind == "data"]
    G = unit_stage(E, ["EEG Fpz-Cz"], "th_snr_db_hat")
    x = (G.N2 - G.W).dropna()
    snr["Sleep-EDF N2-W, Fpz-Cz"] = dict(n=int(len(x)), median=float(x.median()), p5=float(x.quantile(.05)), p95=float(x.quantile(.95)), frac_positive=float((x > 0).mean()))
    out["SNR changes (dB)"] = snr
    return out


# (4) cluster tests on standardized effects
def dz_clusters(scalp, adj2):
    out = {}
    chans, adj = tdbrain_adj()
    P = tdbrain_post(["peak", "peak_height", "th_mu_q50", "th_c_q50"])
    P = P[P.kind == "data"]; P["cred"] = credible(P, "alpha")
    for tag, d in (("all", P), ("credible peaks", P[P.cred])):
        w = d.pivot_table(index=["sub", "CH"], columns="task", values="delta")
        D = (w.restEO - w.restEC).unstack("CH").reindex(columns=chans).values
        _, cl = cluster_test(D, adj, n_perm=NPERM, seed=1)
        out[f"TDBRAIN EO-EC delta, {tag}"] = {"t-mass clusters": clusters_list(cl, chans),
                                              "d_z-mass clusters": clusters_list(cluster_dz(D, adj), chans),
                                              "participants per channel": [int(np.sum(np.isfinite(D), 0).min()), int(np.sum(np.isfinite(D), 0).max())]}
    S = sleep_table("anphy_sigma", ["peak", "peak_height", "th_mu_q50", "th_c_q50"])
    S = S[S.kind == "data"]; S["cred"] = credible(S, "sigma")
    for tag, d in (("all", S), ("credible peaks", S[S.cred])):
        g = d[d.ch.isin(scalp)].groupby(["rec", "ch", "stage"]).delta.median().unstack()
        D = (g.N2 - g.W).unstack("ch").reindex(columns=scalp).values
        _, cl = cluster_test(D, adj2, n_perm=NPERM, seed=3)
        out[f"ANPHY N2-W delta, {tag}"] = {"t-mass clusters": clusters_list(cl, scalp),
                                           "d_z-mass clusters": clusters_list(cluster_dz(D, adj2, seed=3), scalp),
                                           "nights per channel": [int(np.sum(np.isfinite(D), 0).min()), int(np.sum(np.isfinite(D), 0).max())]}
    return out


# (5) state-axis alignment
def vec(D, x=None):
    if x is None:
        return D.mean(0)
    Z = np.column_stack([np.ones(len(x)), x])
    return np.linalg.lstsq(Z, D, rcond=None)[0][0]


def vboot(D, x=None):
    ok = np.isfinite(D).all(1) & (np.isfinite(x) if x is not None else True)
    D = D[ok]; x = x[ok] if x is not None else None
    n = len(D)
    bs = np.array([vec(D[i], None if x is None else x[i]) for i in (rng.integers(0, n, n) for _ in range(NB))])
    return vec(D, x), bs, n


def cosv(a, b):
    return np.sum(a * b, -1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))


def displacement(G, a, b):
    """Per-participant displacement b - a in (mu, c) and the SNR change; G indexed by unit with columns (k, stage)."""
    D = np.column_stack([G[(k, b)] - G[(k, a)] for k in MC])
    return D, (G[("th_snr_db_hat", b)] - G[("th_snr_db_hat", a)]).values


def axis_alignment():
    K = MC + ["th_snr_db_hat"]
    P = tdbrain_post(["peak", "peak_height"] + K)
    P = P[(P.kind == "data") & P.CH.isin(POST6)]; P["cred"] = credible(P, "alpha")
    A = sleep_table("sleep_alpha", ["peak", "peak_height"] + K); A = A[A.kind == "data"]; A["cred"] = credible(A, "alpha100")
    N = sleep_table("anphy_alpha", ["peak", "peak_height"] + K); N = N[N.kind == "data"]; N["cred"] = credible(N, "alpha")
    variants = {"all": lambda d: d, "credible peaks": lambda d: d[d.cred], "within support": lambda d: d[d.covered],
                "credible peaks within support": lambda d: d[d.cred & d.covered]}
    out = {}
    for vname, sel in list(variants.items()) + [("at zero SNR change", lambda d: d)]:
        snr = vname == "at zero SNR change"
        p = sel(P)
        G = p.groupby(["sub", "task"])[K].mean().unstack()
        D, x = displacement(G, "restEC", "restEO")
        V = {"eye opening (TDBRAIN)": vboot(D, x if snr else None)}
        for name, tab, chans, agg in (("wake to N1, Sleep-EDF Pz-Oz", A, ["EEG Pz-Oz"], "mean"), ("wake to N1, Sleep-EDF Fpz-Cz", A, ["EEG Fpz-Cz"], "mean"),
                                      ("wake to N1, ANPHY posterior", N, ANPHY_POST, "median"), ("wake to N1, ANPHY Pz-Oz", N, ["PZ-OZ"], "median")):
            d = sel(tab)
            G = pd.concat({k: unit_stage(d, chans, k, agg) for k in K}, axis=1)
            D, x = displacement(G, "W", "N1")
            V[name] = vboot(D, x if snr else None)
        u, ub = V["eye opening (TDBRAIN)"][0], V["eye opening (TDBRAIN)"][1]
        u, ub = u / np.linalg.norm(u), ub / np.linalg.norm(ub, axis=1, keepdims=True)
        perp = lambda v: np.stack([-v[..., 1], v[..., 0]], -1)
        r = {}
        for name, (est, bs, n) in V.items():
            e = dict(n=n, d_mu=float(est[0]), d_c=float(est[1]))
            if name != "eye opening (TDBRAIN)":
                cb = cosv(bs, ub)
                e.update(cos_eye_opening=float(cosv(est, u)), cos_ci=[float(np.percentile(cb, 2.5)), float(np.percentile(cb, 97.5))],
                         along=float(est @ u), orth=float(est @ perp(u)),
                         orth_ci=[float(np.percentile(np.sum(bs * perp(ub), 1), 2.5)), float(np.percentile(np.sum(bs * perp(ub), 1), 97.5))])
            r[name] = e
        for a, b in (("wake to N1, Sleep-EDF Pz-Oz", "wake to N1, ANPHY posterior"), ("wake to N1, Sleep-EDF Pz-Oz", "wake to N1, ANPHY Pz-Oz")):
            cb = cosv(V[a][1], V[b][1])
            r[f"cos({a}, {b})"] = dict(cos=float(cosv(V[a][0], V[b][0])), ci=[float(np.percentile(cb, 2.5)), float(np.percentile(cb, 97.5))])
        out[vname] = r
    return out


def external_covariates():
    R = pd.read_csv("results/nf_tdbrain_reclevel.csv")
    H = R[R.group == "HEALTHY"].pivot_table(index="sub", columns="task", values=MC).dropna()
    D = np.column_stack([H[(k, "restEO")] - H[(k, "restEC")] for k in MC])
    est, bs, _ = vboot(D)
    u, ub = est / np.linalg.norm(est), bs / np.linalg.norm(bs, axis=1, keepdims=True)
    out = {}
    for name, f, cf, chans, grp in (("Mumtaz", "results/post_mumtaz.csv", "results/coverage_mumtaz.csv", POST_M, "group"),
                                    ("ds003478", "results/post_ds003478.csv", "results/coverage_ds003478.csv", POST_D, "BDI")):
        P = pd.read_csv(f, usecols=COLS + [grp])
        keys = ["sub", grp] if grp == "BDI" else ["group", "sub"]
        d, _ = rec_table(P, pd.read_csv(cf).cov_pct.values < 95, chans, keys)
        d = d.reset_index(drop=True)
        a, b = ((d.BDI >= 13).values, (d.BDI <= 6).values) if grp == "BDI" else ((d.group == "MDD").values, (d.group == "HEALTHY").values)
        r = {"group difference in spectral covariates (standardized)": {
            c: dict(diff_sd=float((d[c][a].mean() - d[c][b].mean()) / d[c][a | b].std()), p_mannwhitney=float(mannwhitneyu(d[c][a].dropna(), d[c][b].dropna()).pvalue))
            for c in SPEC}}
        for cov in ([], ["rho"], ["peak"], ["peak_height"], SPEC):
            X = np.column_stack([np.ones(len(d))] + [((d[c] - d[c].mean()) / d[c].std()).values for c in cov])
            Y = d[MC].values
            ok = np.isfinite(X).all(1) & np.isfinite(Y).all(1)
            ia, ib = np.where(ok & a)[0], np.where(ok & b)[0]

            def contrast(ia, ib):
                beta = np.linalg.lstsq(X[ib], Y[ib], rcond=None)[0]
                return (Y[ia] - X[ia] @ beta).mean(0) - (Y[ib] - X[ib] @ beta).mean(0)
            e = contrast(ia, ib)
            reps = np.array([contrast(rng.choice(ia, len(ia)), rng.choice(ib, len(ib))) for _ in range(NB)])
            al = np.sum(reps * ub, 1)
            r["none" if not cov else " + ".join(cov)] = dict(n=f"{len(ia)}/{len(ib)}", along=float(e @ u),
                                                           ci=[float(np.percentile(al, 2.5)), float(np.percentile(al, 97.5))],
                                                           p=float(min(1.0, max(1 / NB, 2 * min(np.mean(al <= 0), np.mean(al >= 0))))))
        out[name] = r
    return out


# (6) mu_t
def mu_tilde():
    out = {}
    P = pd.read_csv("results/post_tdbrain.csv", usecols=["sub", "ses", "task", "ch", "kind", "th_mu_q50", "th_c_q50"])
    P = P[P.ch.str.upper().isin(POST4) & P.kind.isin(["data", "iaaft0"])]
    P["delta"] = fold_delta(P.th_mu_q50.values, P.th_c_q50.values)
    k = ["sub", "ses", "task", "ch"]
    R = P[P.kind == "data"].set_index(k)[["th_mu_q50", "th_c_q50", "delta"]]
    R = R.loc[R.index.intersection(P[P.kind == "iaaft0"].set_index(k).index)].groupby(level=["sub", "ses", "task"]).mean()
    H = R.xs("ses-1", level="ses").reset_index()
    H = H[H["sub"].isin(healthy_adults())]
    W = H.pivot_table(index="sub", columns="task", values=["th_mu_q50", "th_c_q50", "delta"]).dropna()
    out["EO-EC, O1/Oz/O2/Pz mean"] = {m: dzci((W[(m, "restEO")] - W[(m, "restEC")]).values) for m in ("th_mu_q50", "th_c_q50", "delta")}
    rel = {}
    for task in ("restEC", "restEO"):
        S = R.xs(task, level="task").reset_index().pivot_table(index="sub", columns="ses", values="th_mu_q50")
        ab = S[["ses-1", "ses-2"]].dropna().values
        rel[task] = dict(n=int(len(ab)), icc=icc21(ab[:, 0], ab[:, 1]), ci=ci(lambda i: icc21(ab[i, 0], ab[i, 1]), len(ab)))
    out["test-retest ICC of mu_t"] = rel
    for t, chf in (("anphy_sigma", None), ("sleep_sigma", "EEG Fpz-Cz")):
        S = sleep_table(t, ["th_mu_q50", "th_c_q50"])
        S = S[(S.kind == "data") & S.stage.isin(["W", "N2"])]
        S = S[(S.ch == chf) if chf else ~S.ch.str.contains("-")]
        G = S.groupby(["unit", "stage"])[["th_mu_q50", "th_c_q50", "delta"]].mean().unstack()
        out[f"N2-W, {t}"] = {m: dzci((G[(m, "N2")] - G[(m, "W")]).values) for m in ("th_mu_q50", "th_c_q50", "delta")}
    # channel-wise decomposition of the eye-opening change in delta: delta = mu_t + (c_t^2/4 for c_t > 0)
    Q = tdbrain_post(["th_mu_q50", "th_c_q50"])
    Q = Q[Q.kind == "data"].copy(); Q["cterm"] = Q.delta - Q.th_mu_q50
    w = Q.pivot_table(index=["sub", "CH"], columns="task", values=["th_mu_q50", "cterm", "delta"]).dropna()
    dd = pd.DataFrame({m: w[(m, "restEO")] - w[(m, "restEC")] for m in ("th_mu_q50", "cterm", "delta")})
    ch = dd.groupby(level="CH").agg(["mean", lambda x: x.mean() / x.std(ddof=1)])
    ch.columns = [f"{a}_{'mean' if b == 'mean' else 'dz'}" for a, b in ch.columns]
    out["EO-EC by channel (mean change; d_z)"] = ch.round(3).to_dict(orient="index")
    out["EO-EC by region (mean of channel means)"] = {
        g: {m: float(ch.loc[[c for c in cs if c in ch.index], f"{m}_mean"].mean()) for m in ("th_mu_q50", "cterm", "delta")}
        for g, cs in REGIONS.items()}
    return out


# (7) derivations in the sleep data
def derivations():
    cols = ["th_mu_q50", "th_c_q50", "log_t0_q50", "kurt", "cv", "peak", "peak_height", "rho", "th_snr_db_hat"]
    out = {}
    for t, sets in (("anphy_sigma", {"scalp mean": None, "FPZ-CZ": ["FPZ-CZ"], "PZ-OZ": ["PZ-OZ"]}),
                    ("sleep_sigma", {"Fpz-Cz": ["EEG Fpz-Cz"], "Pz-Oz": ["EEG Pz-Oz"]})):
        S = sleep_table(t, cols)
        S = S[(S.kind == "data") & S.stage.isin(["W", "N2"])]; S["cred"] = credible(S, "sigma").astype(float)
        for name, chans in sets.items():
            d = S[~S.ch.str.contains("-")] if chans is None else S[S.ch.isin(chans)]
            r = {}
            for tag, dd in (("all pieces", d), ("credible peaks", d[d.cred == 1])):
                G = dd.groupby(["unit", "stage"])[["delta", "th_mu_q50", "th_c_q50", "log_t0_q50", "kurt", "cv"]].mean().unstack()
                r[f"N2-W d_z, {tag}"] = {m: dzci((G[(m, "N2")] - G[(m, "W")]).values) for m in ("delta", "th_mu_q50", "th_c_q50", "log_t0_q50", "kurt", "cv")}
            G = d.groupby(["unit", "stage"])[["peak", "peak_height", "rho", "th_snr_db_hat", "cred", "log_t0_q50", "kurt"]].median().unstack()
            r["participant medians by stage"] = {s: {m: float(G[(m, s)].median()) for m in ("peak", "peak_height", "rho", "th_snr_db_hat", "log_t0_q50", "kurt")}
                                                 for s in ("W", "N2")}
            r["fraction of pieces with a credible peak"] = {s: float(d[d.stage == s].cred.mean()) for s in ("W", "N2")}
            out[f"{t} {name}"] = r
    return out


# (8) DFA exponent
def dfa_benchmark(scalp):
    out = {}
    fs = glob.glob("results/tdbrain_checks/*.csv")
    if fs:
        B = pd.concat([pd.read_csv(f) for f in fs if open(f).readline().strip()], ignore_index=True)
        B["CH"] = B.ch.str.upper()
        B = B[B.CH.isin(POST4)]
        M = B.groupby(["sub", "ses", "task", "kind"]).dfa.mean().unstack("kind")
        H = M.xs("ses-1", level="ses")
        H = H[H.index.get_level_values("sub").isin(healthy_adults())]
        r = {}
        for task in ("restEC", "restEO"):
            x = H.xs(task, level="task")[["data", "twin_matched"]].dropna()
            r[task] = dict(n=int(len(x)), median_eeg=float(x.data.median()), median_twin=float(x.twin_matched.median()),
                           frac_eeg_higher=float((x.data > x.twin_matched).mean()), p_wilcoxon=float(wilcoxon(x.data, x.twin_matched).pvalue))
        w = H.data.unstack("task").dropna()
        r["EO-EC d_z, EEG"] = dzci((w.restEO - w.restEC).values)
        w2 = (H.data - H.twin_matched).unstack("task").dropna()
        r["EO-EC d_z, EEG minus twin"] = dzci((w2.restEO - w2.restEC).values)
        rel = {}
        for task in ("restEC", "restEO"):
            S = M.data.xs(task, level="task").unstack("ses")
            if "ses-2" in S:
                ab = S[["ses-1", "ses-2"]].dropna().values
                rel[task] = dict(n=int(len(ab)), icc=icc21(ab[:, 0], ab[:, 1]), ci=ci(lambda i: icc21(ab[i, 0], ab[i, 1]), len(ab)))
        r["test-retest ICC(2,1)"] = rel
        P = tdbrain_post(["th_mu_q50", "th_c_q50", "log_t0_q50"])
        P = P[(P.kind == "data") & P.CH.isin(POST4)].groupby(["sub", "task"])[["delta", "th_c_q50", "log_t0_q50"]].mean()
        J = H.data.rename("dfa").to_frame().join(P, how="inner")
        R4 = tdbrain_table().xs("ses-1", level="ses")
        J = J.join(R4[DESC], how="left")
        r["spearman with dfa (healthy, both conditions)"] = {m: float(spearmanr(J.dfa, J[m], nan_policy="omit")[0]) for m in ["delta", "th_c_q50", "log_t0_q50"] + DESC}
        out["TDBRAIN alpha, O1/Oz/O2/Pz"] = r
    for ds, chans in (("anphy", scalp), ("sleep", ["EEG Fpz-Cz"])):
        fs = glob.glob(f"results/sleep_checks_{ds}/*.csv")
        if not fs:
            continue
        S = pd.concat([pd.read_csv(f) for f in fs if open(f).readline().strip()], ignore_index=True)
        S["unit"] = S.rec.str[:5] if ds == "sleep" else S.rec
        D, T = S[S.kind == "data"], S[S.kind == "iaaft0"]
        r = {}
        for col in ("dfa_peak", "dfa_fixed"):
            G = unit_stage(D, chans, col)
            r[f"N2-W d_z, {col}"] = dzci((G.N2 - G.W).values)
            r[f"median by stage, {col}"] = {s: float(G[s].median()) for s in ("W", "N2")}
        Dm = D.merge(T[["rec", "ch", "stage", "chunk"]], on=["rec", "ch", "stage", "chunk"])
        Ge, Gt = unit_stage(Dm, chans, "dfa_fixed"), unit_stage(T, chans, "dfa_fixed")
        for s in ("W", "N2"):
            x = pd.concat([Ge[s], Gt[s]], axis=1, keys=["e", "t"]).dropna()
            r[f"EEG minus twin, fixed band, {s}"] = dict(n=int(len(x)), median_eeg=float(x.e.median()), median_twin=float(x.t.median()),
                                                        p_wilcoxon=float(wilcoxon(x.e, x.t).pvalue))
        out[f"{ds} spindle band"] = r
    return out


# (9) bursts
def cross_session(pred_file, S1):
    b = pd.concat([pd.read_csv(f) for f in glob.glob("results/bursts_retest/*.csv") if open(f).readline().strip()])
    key = ["sub", "task", "ch"]
    obs2 = b[(b.kind == "data") & (b.ses == "ses-2")].set_index(key)[STATS]
    obs1 = b[(b.kind == "data") & (b.ses == "ses-1")].set_index(key)[STATS]
    tw1 = b[(b.kind == "twin") & (b.ses == "ses-1")].set_index(key)[STATS]
    pred = pd.read_csv(pred_file).groupby(key)[STATS].mean()
    idx = obs2.index.intersection(obs1.index).intersection(tw1.index).intersection(pred.index)
    L = pd.concat({"obs2": obs2.loc[idx], "obs1": obs1.loc[idx], "twin1": tw1.loc[idx], "pred1": pred.loc[idx]}, axis=1)
    L = L.groupby(level=["sub", "task"]).mean()
    L.columns = [f"{a}|{b}" for a, b in L.columns]
    out = {}
    ref = DESC + [m + "_ref" for m in DESC]
    for task in ("restEC", "restEO"):
        r = L.xs(task, level="task").join(S1.xs(task, level="task")[ref], how="inner")
        for s in STATS:
            q = r[[f"{a}|{s}" for a in ("obs2", "obs1", "twin1", "pred1")] + ref].dropna()
            o2, o1, tw, pr = (q[f"{a}|{s}"].rank().values for a in ("obs2", "obs1", "twin1", "pred1"))
            n = len(q)
            Z = np.column_stack([np.ones(n), o1, tw] + [q[m].rank().values for m in ref] + [pr])
            beta, *_ = np.linalg.lstsq(Z, o2, rcond=None)
            res = o2 - Z @ beta
            tval = beta[-1] / np.sqrt(res @ res / (n - Z.shape[1]) * np.linalg.inv(Z.T @ Z)[-1, -1])
            out[f"{task}|{s}"] = dict(n=n, rho_pred1=float(spearmanr(o2, pr)[0]), rho_obs1=float(spearmanr(o2, o1)[0]),
                                      t_pred1_beyond_obs1_twin_desc=float(tval))
    return out


def bursts():
    out = {}
    key = ["sub", "task", "ch"]
    b = pd.concat([pd.read_csv(f) for f in glob.glob("results/bursts_tdbrain/*.csv") if open(f).readline().strip()])
    obs, twin = b[b.kind == "data"].set_index(key)[STATS], b[b.kind == "twin"].set_index(key)[STATS]
    med = pd.read_csv("results/bursts_ppc.csv").groupby(key)[STATS].mean()
    post = pd.read_csv("results/post_tdbrain.csv", usecols=["sub", "ses", "task", "ch", "kind", "th_mu_q50", "th_c_q50", "log_t0_q50",
                                                             "log_t0_q5", "log_t0_q95", "th_snr_db_hat", "peak_height", "rho"])
    post["cov_pct"] = pd.read_csv("results/coverage_tdbrain.csv").cov_pct.values
    post = post[(post.kind == "data") & (post.ses == "ses-1")].set_index(key)
    post["delta"] = fold_delta(post.th_mu_q50, post.th_c_q50); post["t0_width"] = post.log_t0_q95 - post.log_t0_q5
    preds = {"median coordinates": med}
    if os.path.exists("results/bursts_ppc_post.csv"):
        pp = pd.read_csv("results/bursts_ppc_post.csv")
        preds["posterior draws"] = pp.groupby(key)[STATS].mean()
        lo, hi = pp.groupby(key)[STATS].quantile(0.05), pp.groupby(key)[STATS].quantile(0.95)
    for name, pr in preds.items():
        idx = obs.index.intersection(pr.index).intersection(twin.index)
        r = {}
        for s in ("b_dur_mean", "b2_dur_mean", "b_frac", "b2_frac", "b_rate"):
            o, p, tw = obs.loc[idx, s], pr.loc[idx, s], twin.loc[idx, s]
            ok = o.notna() & p.notna() & tw.notna()
            rec = pd.concat([o[ok], p[ok], tw[ok]], axis=1, keys=["o", "p", "t"]).groupby(level=["sub", "task"]).mean()
            e = {}
            for task in ("restEC", "restEO"):
                x = rec.xs(task, level="task")
                e[task] = dict(n=int(len(x)), median_rel_error=float(((x.p - x.o) / x.o).median()), rho_obs_pred=float(spearmanr(x.o, x.p)[0]),
                               frac_pred_closer_than_twin=float(((x.o - x.p).abs() < (x.o - x.t).abs()).mean()))
            if name == "posterior draws":
                inside = (o[ok] >= lo.loc[idx, s][ok]) & (o[ok] <= hi.loc[idx, s][ok])
                e["fraction of channel recordings inside the 90% predictive interval"] = float(inside.mean())
            r[s] = e
        out[f"prediction from {name}"] = r
    # what the relative burst-duration error of the median prediction depends on (channel level)
    idx = obs.index.intersection(med.index).intersection(post.index)
    rel = ((med.loc[idx, "b_dur_mean"] - obs.loc[idx, "b_dur_mean"]) / obs.loc[idx, "b_dur_mean"]).rename("rel")
    J = rel.to_frame().join(post[["delta", "th_c_q50", "log_t0_q50", "t0_width", "th_snr_db_hat", "peak_height", "rho", "cov_pct"]]).dropna()
    out["relative duration error (median prediction): spearman with"] = {c: float(spearmanr(J.rel, J[c])[0]) for c in J.columns if c != "rel"}
    out["relative duration error by support"] = {lab: dict(n=int(m.sum()), median=float(J.rel[m].median()))
                                                 for lab, m in (("inside", J.cov_pct < 95), ("outside", J.cov_pct >= 95))}
    q = pd.qcut(J.th_snr_db_hat, 4, labels=False)
    out["relative duration error by SNR quartile"] = {int(k): float(J.rel[q == k].median()) for k in range(4)}
    if os.path.exists("results/bursts_ppc_recovery.csv"):
        V = pd.read_csv("results/bursts_ppc_recovery.csv")
        G = V.groupby(["sim", "kind"])[["b_dur_mean", "b2_dur_mean", "b_frac", "b2_frac"]].mean().unstack("kind")
        out["simulations with known parameters: prediction at posterior medians relative to the true parameters"] = {
            s: dict(n=int(G[s].dropna().shape[0]), median_rel_error=float(((G[(s, "median")] - G[(s, "true")]) / G[(s, "true")]).median()),
                    rho=float(spearmanr(G[(s, "true")], G[(s, "median")], nan_policy="omit")[0])) for s in ("b_dur_mean", "b2_dur_mean", "b_frac", "b2_frac")}
    S1 = tdbrain_table().xs("ses-1", level="ses")
    out["cross-session, median coordinates"] = cross_session("results/bursts_ppc_retest.csv", S1)
    if os.path.exists("results/bursts_ppc_post_retest.csv"):
        out["cross-session, posterior draws"] = cross_session("results/bursts_ppc_post_retest.csv", S1)
    return out


def main():
    scalp, adj2 = anphy_scalp()
    out = {}
    steps = [("twins with re-estimated peak and background", twins_reestimated),
             ("fixed spindle band", lambda: fixed_band(scalp, adj2)),
             ("credible peaks and SNR changes", lambda: credible_and_snr(scalp)),
             ("standardized cluster tests", lambda: dz_clusters(scalp, adj2)),
             ("state-axis alignment", axis_alignment),
             ("external cohorts, single spectral covariates", external_covariates),
             ("mu_t", mu_tilde),
             ("sleep derivations", derivations),
             ("DFA", lambda: dfa_benchmark(scalp)),
             ("bursts", bursts)]
    for name, f in steps:
        out[name] = f()
        print(f"== {name}\n" + json.dumps(out[name], indent=1, default=str)[:6000], flush=True)
        json.dump(out, open("results/sensitivity_checks.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()

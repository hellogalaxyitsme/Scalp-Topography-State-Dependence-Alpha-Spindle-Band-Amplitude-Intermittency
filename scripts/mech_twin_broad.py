"""Mechanism classifier with a broader intrinsic comparator. Besides intrinsic normal-form dynamics with additive
noise, the intrinsic class now includes the same dynamics with additive plus state-dependent (multiplicative) noise
(sbi_twinbank.py class 'mult'), the noise structure required by the canonical multistability model of Freyer et al.
(2012). Three questions:
(A) How does the original classifier (additive intrinsic vs fast modulation, mech_twin.py) score the
    multiplicative-noise class?
(B) A classifier trained on intrinsic (additive + multiplicative) vs fast modulation: held-out performance,
    empirical negative controls (twin minus an independent twin) in TDBRAIN, ANPHY-Sleep and Sleep-EDF, and the
    EEG (each piece minus its twin), including the state dependence of the score.
(C) A three-class classifier (additive / multiplicative / fast), and domain classifiers between near-fold EEG and
    each near-fold simulated class.
TDBRAIN: healthy reference sample only (adults, first session; 179 eyes closed, 175 eyes open).
usage: python scripts/mech_twin_broad.py  -> results/mech_twin_broad.json, mech_twin_broad_points.csv, mechtwb_<table>.csv"""
import glob, json, joblib, numpy as np, pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from sbi_fit import predict
from ampint.nf import fold_delta
from mech_twin import load, differenced, domain_auc, DIFF, CTX, NEAR, TABLES, HGB

COLS = ["d_" + f for f in DIFF] + CTX
TWIN2 = {"anphy_alpha": ("results/twin2_anphy", "alpha"), "anphy_sigma": ("results/twin2_anphy", "sigma250"),
         "sleep_alpha": ("results/twin2_sleep", "alpha"), "sleep_sigma": ("results/twin2_sleep", "sigma250")}
STATE = {"tdbrain": ("sub", "task", "restEC", "restEO"), "anphy_sigma": ("rec", "stage", "W", "N2"),
         "sleep_sigma": ("rec", "stage", "W", "N2"), "anphy_alpha": ("rec", "stage", "W", "N1"), "sleep_alpha": ("rec", "stage", "W", "N1")}


R0 = pd.read_csv("results/nf_tdbrain_reclevel.csv")
HEALTHY_SUBS = set(R0.loc[R0.group == "HEALTHY", "sub"])


def auc(a, b):
    return float(roc_auc_score(np.r_[np.zeros(len(a)), np.ones(len(b))], np.r_[a, b]))


def twin2(t):
    d, band = TWIN2[t]
    fs = [f for f in glob.glob(f"{d}/*.csv") if open(f).readline().strip()]
    T = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    return T[T.band == band].drop(columns=["band"])


def main():
    rep, figrows = {}, []
    for cfg, tables in TABLES.items():
        reg = joblib.load(f"results/sbi_{cfg}_full.joblib")
        C = {k: load(cfg, k) for k in ("intrinsic", "mult", "fast")}
        for k, d in C.items():
            d["split"] = np.random.default_rng(0 if k != "mult" else 7).choice(["tr", "cal", "te"], len(d), p=[0.8, 0.1, 0.1])
            pr = predict(reg, d)
            d["d_hat"] = fold_delta(pr.th_mu_q50.values, pr.th_c_q50.values); d["d_true"] = fold_delta(d.th_mu.values, d.th_c.values)
        I, M, F = C["intrinsic"], C["mult"], C["fast"]
        te = {k: d[d.split == "te"].copy() for k, d in C.items()}
        r = dict(n={k: int(len(d)) for k, d in C.items()})
        # (A) original classifier on the multiplicative class
        A = joblib.load(f"results/mechtw_{cfg}.joblib")
        pA = lambda d: A["iso"].predict(A["clf"].predict_proba(d[A["cols"]].values)[:, 1])
        for k in te:
            te[k]["pA"] = pA(te[k])
        nm = te["mult"].d_hat.between(*NEAR)
        r["(A) original classifier"] = {"mean p_fast, multiplicative": float(te["mult"].pA.mean()),
                                        "mean p_fast, multiplicative near-fold estimate": float(te["mult"].pA[nm].mean()),
                                        "fraction > 0.5, multiplicative near-fold estimate": float((te["mult"].pA[nm] > 0.5).mean()),
                                        "auc multiplicative vs fast": auc(te["mult"].pA, te["fast"].pA)}
        # (B) intrinsic (additive + multiplicative) vs fast
        S = lambda s: (np.vstack([I[I.split == s][COLS].values, M[M.split == s][COLS].values, F[F.split == s][COLS].values]),
                       np.r_[np.zeros((I.split == s).sum() + (M.split == s).sum()), np.ones((F.split == s).sum())])
        clf = HistGradientBoostingClassifier(**HGB).fit(*S("tr"))
        Xc, yc = S("cal"); iso = IsotonicRegression(out_of_bounds="clip").fit(clf.predict_proba(Xc)[:, 1], yc)
        pB = lambda d: iso.predict(clf.predict_proba(d[COLS].values)[:, 1])
        joblib.dump(dict(clf=clf, iso=iso, cols=COLS), f"results/mechtwb_{cfg}.joblib")
        for k in te:
            te[k]["pB"] = pB(te[k])
        near = {k: te[k].d_hat.between(*NEAR) for k in te}
        intr = pd.concat([te["intrinsic"], te["mult"]]); ni = intr.d_hat.between(*NEAR)
        b = {"auc intrinsic (both) vs fast": auc(intr.pB, te["fast"].pB),
             "auc additive vs fast": auc(te["intrinsic"].pB, te["fast"].pB),
             "auc multiplicative vs fast": auc(te["mult"].pB, te["fast"].pB),
             "auc near-fold estimates": auc(intr.pB[ni], te["fast"].pB[near["fast"]]),
             "auc multiplicative near-fold vs fast near-fold": auc(te["mult"].pB[near["mult"]], te["fast"].pB[near["fast"]]),
             "false-positive rate near fold (intrinsic, both)": float((intr.pB[ni] > 0.5).mean()),
             "false-positive rate near fold (multiplicative)": float((te["mult"].pB[near["mult"]] > 0.5).mean()),
             "recall near fold (fast)": float((te["fast"].pB[near["fast"]] > 0.5).mean()),
             "mean p: deep additive / near-fold additive / near-fold multiplicative / near-fold fast":
                 [float(te["intrinsic"].pB[te["intrinsic"].d_true < -3].mean()), float(te["intrinsic"].pB[te["intrinsic"].d_true.between(*NEAR)].mean()),
                  float(te["mult"].pB[near["mult"]].mean()), float(te["fast"].pB[near["fast"]].mean())]}
        ft = te["fast"]; ft["sd_bin"] = pd.cut(ft.mod_sd, [0.5, 1.5, 2.5, 3.5]); ft["tau_bin"] = pd.cut(ft.mod_tau, [0.2, 0.6, 1.8, 5.0])
        b["recall by modulation"] = {f"sd {a} tau {c}": float((g.pB > 0.5).mean()) for (a, c), g in ft.groupby(["sd_bin", "tau_bin"], observed=True)}
        r["(B) intrinsic (additive + multiplicative) vs fast"] = b
        for lab, d, m in (("deep focus, additive", te["intrinsic"], te["intrinsic"].d_true < -3),
                          ("near-fold intrinsic, additive", te["intrinsic"], te["intrinsic"].d_true.between(*NEAR)),
                          ("near-fold estimate, multiplicative", te["mult"], near["mult"]),
                          ("near-fold estimate, fast modulation", te["fast"], near["fast"])):
            figrows += [dict(cfg=cfg, category=lab, p=v) for v in d.pB[m].values]
        # (C) three classes
        lab3 = lambda s: np.r_[np.zeros((I.split == s).sum()), np.ones((M.split == s).sum()), 2 * np.ones((F.split == s).sum())]
        X3 = lambda s: np.vstack([I[I.split == s][COLS].values, M[M.split == s][COLS].values, F[F.split == s][COLS].values])
        c3 = HistGradientBoostingClassifier(**HGB).fit(X3("tr"), lab3("tr"))
        P3 = c3.predict_proba(X3("te")); y3 = lab3("te")
        r["(C) three-class confusion (rows true additive/multiplicative/fast)"] = pd.crosstab(y3, P3.argmax(1), normalize="index").round(3).values.tolist()
        deep = I[I.d_true < -2.5]
        E = {}
        for t, keys in tables.items():
            P = pd.read_csv(f"results/post_{t}.csv")
            if t == "tdbrain":   # healthy reference sample: adults, first session (as in the eye-opening analyses)
                P = P[P["sub"].isin(HEALTHY_SUBS) & (P.ses == "ses-1")]
            P["delta"] = fold_delta(P.th_mu_q50.values, P.th_c_q50.values)
            D = differenced(P[P.kind == "data"], P[P.kind == "iaaft0"], keys)
            D["p_fast"] = pB(D); D[["p_add", "p_mult", "p_fast3"]] = c3.predict_proba(D[COLS].values)
            D[keys + ["delta", "p_fast", "p_add", "p_mult", "p_fast3"]].to_csv(f"results/mechtwb_{t}.csv", index=False)
            nr = D.delta > -1
            e = dict(n=int(len(D)), p_fast=float(D.p_fast.mean()),
                     near=dict(n=int(nr.sum()), p_fast=float(D.p_fast[nr].mean()), frac_gt_0_5=float((D.p_fast[nr] > 0.5).mean()),
                               three_class_mean=[float(D[c][nr].mean()) for c in ("p_add", "p_mult", "p_fast3")]))
            dcols = ["d_" + f for f in DIFF]
            e["domain auc, near-fold EEG vs near-fold simulations"] = {
                k: domain_auc(C[k][C[k].d_hat.between(*NEAR)], D[nr], dcols) for k in ("intrinsic", "mult", "fast")}
            if t in STATE:
                u, s, a, z = STATE[t]
                if u in D:
                    unit = D[u] if t == "tdbrain" else D[u].str[:5] if t.startswith("sleep") else D[u]
                    G = D.assign(unit=unit).groupby(["unit", s]).p_fast.mean().unstack()
                    if a in G and z in G:
                        g = G[[a, z]].dropna()
                        e[f"state {a}->{z} (participant means)"] = dict(n=len(g), mean_a=float(g[a].mean()), mean_b=float(g[z].mean()),
                                                                          p=float(wilcoxon(g[z], g[a]).pvalue))
            ctrl = None
            if (P.kind == "iaaft1").any():
                ctrl = differenced(P[P.kind == "iaaft1"], P[P.kind == "iaaft0"], keys)
            elif t in TWIN2:
                T2 = twin2(t)
                if len(T2):
                    ctrl = differenced(T2, P[P.kind == "iaaft0"], keys)
            if ctrl is not None and len(ctrl):
                ctrl["p_fast"] = pB(ctrl); ctrl["pA"] = pA(ctrl)
                e["negative control (twin - twin)"] = dict(n=int(len(ctrl)), p_fast=float(ctrl.p_fast.mean()),
                                                           frac_gt_0_5=float((ctrl.p_fast > 0.5).mean()),
                                                           p_fast_original_classifier=float(ctrl.pA.mean()),
                                                           domain_auc_vs_simulated_deep_foci=domain_auc(deep, ctrl, dcols))
                figrows += [dict(cfg=cfg, category=f"twin minus twin, {t}", p=v) for v in ctrl.p_fast.values]
            figrows += [dict(cfg=cfg, category=f"EEG near fold, {t}", p=v) for v in D.p_fast[nr].values]
            E[t] = e
        r["empirical"] = E
        rep[cfg] = r
        print(cfg, json.dumps(r, indent=1), flush=True)
    json.dump(rep, open("results/mech_twin_broad.json", "w"), indent=1)
    pd.DataFrame(figrows).to_csv("results/mech_twin_broad_points.csv", index=False)


if __name__ == "__main__":
    main()

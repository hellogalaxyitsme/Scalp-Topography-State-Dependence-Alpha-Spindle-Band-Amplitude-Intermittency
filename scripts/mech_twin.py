"""Mechanism classifier on twin-differenced statistics (sbi_twinbank.py): each of the 91 summaries minus the same
summary of the signal's own IAAFT twin, with noise fraction and duration as context. Differencing removes what a
signal shares with its linear twin (spectral shape, duration, noise level), where the raw-statistic classifier
extrapolates (mech_domain.py).
Validation: held-out discrimination (overall, by number of sources, near the fold); empirical negative control
(TDBRAIN twin minus a second independent twin, which must score like simulated linear foci); domain classifiers
(simulated deep foci against the empirical control, simulations against EEG). Then EEG minus its twin for TDBRAIN,
Sleep-EDF and ANPHY.
usage: python scripts/mech_twin.py"""
import glob, json, joblib, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sbi_fit import predict
from ampint.summ import FEATURES
from ampint.nf import fold_delta

DIFF = [f for f in FEATURES if f not in ("rho", "T")]
CTX = ["rho", "T"]
NEAR = (-1.5, 1.0)
TABLES = {"alpha": {"tdbrain": ["sub", "ses", "task", "ch"], "sleep_alpha": ["rec", "ch", "stage", "chunk"],
                    "anphy_alpha": ["rec", "ch", "stage", "chunk"]},
          "sigma250": {"sleep_sigma": ["rec", "ch", "stage", "chunk"], "anphy_sigma": ["rec", "ch", "stage", "chunk"]}}
HGB = dict(max_iter=800, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=40, l2_regularization=1.0, random_state=0)


def load(cfg, cls):
    fs = sorted(glob.glob(f"results/twinbank_{cfg}_{cls}/chunk_*.npz"))
    d = pd.concat([pd.DataFrame(dict(np.load(f, allow_pickle=True))) for f in fs], ignore_index=True)
    for f in DIFF:
        d["d_" + f] = d[f] - d["tw_" + f]
    return d.replace([np.inf, -np.inf], np.nan).dropna(subset=["d_" + f for f in DIFF] + CTX).reset_index(drop=True)


def differenced(a, b, keys):
    """Rows of a minus matched rows of b (same unit), context from a."""
    m = a.merge(b[keys + DIFF], on=keys, suffixes=("", "_tw"))
    for f in DIFF:
        m["d_" + f] = m[f] - m[f + "_tw"]
    return m.replace([np.inf, -np.inf], np.nan).dropna(subset=["d_" + f for f in DIFF] + CTX).reset_index(drop=True)


def domain_auc(A, B, cols, n=6000):
    n = min(n, len(A), len(B))
    X = np.vstack([A[cols].sample(n, random_state=1).values, B[cols].sample(n, random_state=1).values])
    y = np.r_[np.zeros(n), np.ones(n)]
    p = cross_val_predict(HistGradientBoostingClassifier(max_iter=300, random_state=0), X, y, cv=5, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def main():
    rep = {}
    for cfg, tables in TABLES.items():
        reg = joblib.load(f"results/sbi_{cfg}_full.joblib")
        I, F = load(cfg, "intrinsic"), load(cfg, "fast")
        for d in (I, F):
            d["split"] = np.random.default_rng(0).choice(["tr", "cal", "te"], len(d), p=[0.8, 0.1, 0.1])
            pr = predict(reg, d)
            d["d_hat"] = fold_delta(pr.th_mu_q50.values, pr.th_c_q50.values); d["d_true"] = fold_delta(d.th_mu.values, d.th_c.values)
        cols = ["d_" + f for f in DIFF] + CTX
        S = lambda k: (np.vstack([I[I.split == k][cols].values, F[F.split == k][cols].values]),
                       np.r_[np.zeros((I.split == k).sum()), np.ones((F.split == k).sum())])
        clf = HistGradientBoostingClassifier(**HGB).fit(*S("tr"))
        Xc, yc = S("cal"); iso = IsotonicRegression(out_of_bounds="clip").fit(clf.predict_proba(Xc)[:, 1], yc)
        pf = lambda d: iso.predict(clf.predict_proba(d[cols].values)[:, 1])
        joblib.dump(dict(clf=clf, iso=iso, cols=cols), f"results/mechtw_{cfg}.joblib")
        It, Ft = I[I.split == "te"].copy(), F[F.split == "te"].copy()
        It["p"], Ft["p"] = pf(It), pf(Ft)
        auc = lambda a, b: float(roc_auc_score(np.r_[np.zeros(len(a)), np.ones(len(b))], np.r_[a, b]))
        r = dict(n_train=int((I.split == "tr").sum() + (F.split == "tr").sum()), n_test=f"{len(It)}/{len(Ft)}", auc=auc(It.p, Ft.p))
        r["auc by number of sources"] = {int(k): auc(It.p[It.n_src == k], Ft.p[Ft.n_src == k]) for k in (1, 2, 3)}
        ni, nf = It.d_hat.between(*NEAR), Ft.d_hat.between(*NEAR)
        r["auc, estimate near fold"] = dict(n=f"{int(ni.sum())}/{int(nf.sum())}", auc=auc(It.p[ni], Ft.p[nf]))
        nt = It.d_true.between(*NEAR)
        r["auc, true near-fold intrinsic vs fast with near-fold estimate"] = auc(It.p[nt], Ft.p[nf])
        r["p_fast simulated deep foci (true delta < -3)"] = float(It.p[It.d_true < -3].mean())
        r["p_fast simulated true near-fold intrinsic"] = float(It.p[nt].mean())
        r["p_fast simulated fast, near-fold estimate"] = float(Ft.p[nf].mean())
        r["false positive rate: intrinsic, near-fold estimate"] = float((It.p[ni] > 0.5).mean())
        r["recall: fast, near-fold estimate"] = float((Ft.p[nf] > 0.5).mean())
        Ft["sd_bin"] = pd.cut(Ft.mod_sd, [0.5, 1.5, 2.5, 3.5]); Ft["tau_bin"] = pd.cut(Ft.mod_tau, [0.2, 0.6, 1.8, 5.0])
        r["recall by modulation"] = {f"sd {a} tau {b}": float((g.p > 0.5).mean()) for (a, b), g in Ft.groupby(["sd_bin", "tau_bin"], observed=True)}
        deep = It[It.d_true < -2.5]
        E = {}
        for t, keys in tables.items():
            P = pd.read_csv(f"results/post_{t}.csv")
            P["delta"] = fold_delta(P.th_mu_q50.values, P.th_c_q50.values)
            D = differenced(P[P.kind == "data"], P[P.kind == "iaaft0"], keys); D["p_fast"] = pf(D)
            D[keys + ["delta", "p_fast"]].to_csv(f"results/mechtw_{t}.csv", index=False)
            e = dict(n=int(len(D)), p_fast=float(D.p_fast.mean()), frac_gt_0_5=float((D.p_fast > 0.5).mean()))
            near = D.delta > -1
            e["near-fold rows"] = dict(n=int(near.sum()), p_fast=float(D.p_fast[near].mean()), frac_gt_0_5=float((D.p_fast[near] > 0.5).mean()))
            e["domain auc: simulations vs EEG"] = domain_auc(pd.concat([It, Ft]), D, ["d_" + f for f in DIFF])
            # which simulated mechanism the near-fold EEG resembles more (lower = closer)
            sim_i, sim_f = I[I.d_hat.between(*NEAR)], F[F.d_hat.between(*NEAR)]
            e["domain auc, near-fold EEG vs near-fold simulations"] = dict(
                intrinsic=domain_auc(sim_i, D[near], ["d_" + f for f in DIFF]), fast=domain_auc(sim_f, D[near], ["d_" + f for f in DIFF]))
            if (P.kind == "iaaft1").any():   # negative control: twin minus an independent twin
                C = differenced(P[P.kind == "iaaft1"], P[P.kind == "iaaft0"], keys); C["p_fast"] = pf(C)
                e["negative control (twin - twin)"] = dict(n=int(len(C)), p_fast=float(C.p_fast.mean()),
                                                           frac_gt_0_5=float((C.p_fast > 0.5).mean()),
                                                           domain_auc_vs_simulated_deep_foci=domain_auc(deep, C, ["d_" + f for f in DIFF]))
            E[t] = e
        r["empirical"] = E
        rep[cfg] = r
        print(cfg, json.dumps(r, indent=1), flush=True)
    json.dump(rep, open("results/mech_twin.json", "w"), indent=1)


if __name__ == "__main__":
    main()

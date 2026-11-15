"""Mechanism discrimination: intrinsic normal-form dynamics (simulation bank) against a noisy oscillator whose
Hopf parameter is modulated by a fast Ornstein-Uhlenbeck process (correlation time 0.2-5 s; 'fastmod' bank).
A gradient-boosting classifier on the 91 summary statistics, isotonic-calibrated (equal class priors).
Evaluation on simulations held out from both the classifier and the amortised posterior, overall and in the
confusable subset whose posterior fold distance lies near the fold. Applied to every empirical table.
usage: python scripts/mech_classifier.py"""
import json, joblib, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sbi_fit import load_bank, predict
from ampint.summ import FEATURES
from ampint.nf import fold_delta, regime

TABLES = {"alpha": ["tdbrain", "sleep_alpha", "anphy_alpha", "mumtaz", "ds003478"], "sigma250": ["sleep_sigma", "anphy_sigma"]}
NEAR = (-1.5, 1.0)


def ece(y, p, nb=10):
    b = np.minimum((p * nb).astype(int), nb - 1)
    return float(sum(np.abs(y[b == k].mean() - p[b == k].mean()) * np.mean(b == k) for k in range(nb) if np.any(b == k)))


def main():
    rep = {}
    for cfg, tables in TABLES.items():
        rng = np.random.default_rng(0)
        reg = joblib.load(f"results/sbi_{cfg}_full.joblib")
        B = load_bank(cfg)
        H = pd.read_csv(f"results/sbi_{cfg}_full_heldout.csv")
        assert set(FEATURES) <= set(H.columns) and "seed" in H.columns
        pool = B[~B.seed.isin(set(H.seed))]
        F = load_bank(cfg, "fastmod")
        uf = rng.random(len(F))
        Ftr, Fcal, Fte = F[uf < 0.8], F[(uf >= 0.8) & (uf < 0.9)], F[uf >= 0.9]
        I = pool.sample(len(Ftr) + len(Fcal), random_state=0)
        Itr, Ical = I.iloc[:len(Ftr)], I.iloc[len(Ftr):]
        X = np.vstack([Itr[FEATURES].values, Ftr[FEATURES].values]); y = np.r_[np.zeros(len(Itr)), np.ones(len(Ftr))]
        clf = HistGradientBoostingClassifier(max_iter=800, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=40,
                                             l2_regularization=1.0, random_state=0).fit(X, y)
        Xc = np.vstack([Ical[FEATURES].values, Fcal[FEATURES].values]); yc = np.r_[np.zeros(len(Ical)), np.ones(len(Fcal))]
        iso = IsotonicRegression(out_of_bounds="clip").fit(clf.predict_proba(Xc)[:, 1], yc)
        pf = lambda d: iso.predict(clf.predict_proba(d[FEATURES].values)[:, 1])
        joblib.dump(dict(clf=clf, iso=iso, features=FEATURES), f"results/mech_{cfg}.joblib")
        # held-out evaluation: intrinsic = posterior hold-out set; fast = classifier hold-out set (with posterior estimates)
        It = H.copy(); It["p_fast"] = pf(It); It["d_hat"] = fold_delta(It.th_mu_q50, It.th_c_q50)
        It["d_true"] = fold_delta(It.th_mu, It.th_c)
        if "regime" not in It:
            It["regime"] = [regime(a, b) for a, b in zip(It.th_mu, It.th_c)]
        Ft = pd.concat([Fte.reset_index(drop=True), predict(reg, Fte.reset_index(drop=True))], axis=1)
        Ft["p_fast"] = pf(Ft); Ft["d_hat"] = fold_delta(Ft.th_mu_q50, Ft.th_c_q50)
        yy = np.r_[np.zeros(len(It)), np.ones(len(Ft))]; pp = np.r_[It.p_fast, Ft.p_fast]
        r = dict(n_train=int(len(y)), n_test_intrinsic=int(len(It)), n_test_fast=int(len(Ft)),
                 auc=float(roc_auc_score(yy, pp)), brier=float(brier_score_loss(yy, pp)), ece=ece(yy, pp))
        ni, nf = It.d_hat.between(*NEAR), Ft.d_hat.between(*NEAR)
        r["confusable: posterior delta near fold"] = dict(n_intrinsic=int(ni.sum()), n_fast=int(nf.sum()),
                                                          auc=float(roc_auc_score(np.r_[np.zeros(ni.sum()), np.ones(nf.sum())],
                                                                                  np.r_[It.p_fast[ni], Ft.p_fast[nf]])))
        nt = It.d_true.between(*NEAR)
        r["true near-fold intrinsic vs fast with near-fold estimate"] = dict(
            n_intrinsic=int(nt.sum()), auc=float(roc_auc_score(np.r_[np.zeros(nt.sum()), np.ones(nf.sum())], np.r_[It.p_fast[nt], Ft.p_fast[nf]])),
            median_p_fast_intrinsic=float(It.p_fast[nt].median()), median_p_fast_fast=float(Ft.p_fast[nf].median()))
        # sensitivity of the classifier to modulation strength and timescale
        Ft["sd_bin"] = pd.cut(Ft.mod_sd, [0.5, 1.5, 2.5, 3.5]); Ft["tau_bin"] = pd.cut(Ft.mod_tau, [0.2, 0.6, 1.8, 5.0])
        r["recall_by_modulation"] = {f"sd {a} tau {b}": float((g.p_fast > 0.5).mean())
                                    for (a, b), g in Ft.groupby(["sd_bin", "tau_bin"], observed=True)}
        r["false_positive_rate_intrinsic"] = float((It.p_fast > 0.5).mean())
        r["fpr_intrinsic_by_regime"] = It.groupby("regime").p_fast.apply(lambda v: float((v > 0.5).mean())).to_dict()
        rep[cfg] = r
        print(cfg, json.dumps(r, indent=1, default=str), flush=True)
        for t in tables:
            e = pd.read_csv(f"results/post_{t}.csv", usecols=FEATURES)
            ok = e.replace([np.inf, -np.inf], np.nan).notna().all(1).values
            p = np.full(len(e), np.nan); p[ok] = pf(e[ok])
            pd.DataFrame(dict(p_fast=p)).to_csv(f"results/mech_{t}.csv", index=False)
            print(t, len(e), "rows; mean p_fast", float(np.nanmean(p)), flush=True)
    # source mixing and fast-drive check simulations (sim_alternatives.py)
    S = pd.read_csv("results/sim_alternatives.csv")
    for cfg in ("alpha", "sigma250"):
        m = joblib.load(f"results/mech_{cfg}.joblib"); s = S[S.cfg == cfg]
        S.loc[s.index, "p_fast"] = m["iso"].predict(m["clf"].predict_proba(s[FEATURES].values)[:, 1])
    rep["sim_alternatives"] = dict(
        drive=S[S.case == "drive"].groupby(["cfg", "sd", "tau"]).p_fast.mean().round(3).reset_index().to_dict("records"),
        mixing=S[S.case == "mix"].groupby(["true_delta", "n_src", "df_hz"]).p_fast.mean().round(3).reset_index().to_dict("records"))
    json.dump(rep, open("results/mech_classifier.json", "w"), indent=1, default=str)
    print(json.dumps(rep["sim_alternatives"], indent=1))


if __name__ == "__main__":
    main()

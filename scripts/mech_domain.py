"""Domain check of the mechanism classifier. The linear twins of the EEG are Gaussian processes and should look like
simulated deep (linear) foci; a classifier that assigns them a high fast-modulation probability is extrapolating.
For three feature sets (all 91 statistics; amplitude distribution only; envelope autocorrelation only):
held-out discrimination (overall and near the fold), mean P(fast) for simulated deep foci, empirical twins and
empirical EEG, and the cross-validated AUC of a domain classifier separating simulations from empirical twins.
usage: python scripts/mech_domain.py"""
import json, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sbi_fit import load_bank
from ampint.summ import FEATURES
from ampint.nf import fold_delta

ACF = [f for f in FEATURES if f.startswith("acf") or (f[0] == "a" and f[1:].isdigit())]
SETS = {"all": FEATURES, "amplitude": [f for f in FEATURES if f not in ACF], "autocorrelation": ACF + ["rho", "T"]}
TABLES = {"alpha": {"tdbrain": ["sub", "ses", "task", "ch"], "sleep_alpha": ["rec", "ch", "stage", "chunk"],
                    "anphy_alpha": ["rec", "ch", "stage", "chunk"]},
          "sigma250": {"sleep_sigma": ["rec", "ch", "stage", "chunk"], "anphy_sigma": ["rec", "ch", "stage", "chunk"]}}
NEAR = (-1.5, 1.0)
HGB = dict(max_iter=800, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=40, l2_regularization=1.0, random_state=0)


def paired(name, keys):
    d = pd.read_csv(f"results/post_{name}.csv", usecols=list(set(FEATURES + keys + ["kind", "th_mu_q50", "th_c_q50"])))
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)
    tw = d[d.kind != "data"]; da = d[d.kind == "data"].merge(tw[keys].drop_duplicates(), on=keys)
    return da, tw


def main():
    rep = {}
    for cfg, tables in TABLES.items():
        rng = np.random.default_rng(0)
        B = load_bank(cfg); H = pd.read_csv(f"results/sbi_{cfg}_full_heldout.csv")
        pool = B[~B.seed.isin(set(H.seed))]
        F = load_bank(cfg, "fastmod"); uf = rng.random(len(F))
        Ftr, Fcal, Fte = F[uf < 0.8], F[(uf >= 0.8) & (uf < 0.9)], F[uf >= 0.9]
        I = pool.sample(len(Ftr) + len(Fcal), random_state=0); Itr, Ical = I.iloc[:len(Ftr)], I.iloc[len(Ftr):]
        H["d_true"] = fold_delta(H.th_mu, H.th_c); H["d_hat"] = fold_delta(H.th_mu_q50, H.th_c_q50)
        emp = {t: paired(t, k) for t, k in tables.items()}
        R = {}
        for sname, feats in SETS.items():
            clf = HistGradientBoostingClassifier(**HGB).fit(np.vstack([Itr[feats].values, Ftr[feats].values]),
                                                            np.r_[np.zeros(len(Itr)), np.ones(len(Ftr))])
            iso = IsotonicRegression(out_of_bounds="clip").fit(
                clf.predict_proba(np.vstack([Ical[feats].values, Fcal[feats].values]))[:, 1], np.r_[np.zeros(len(Ical)), np.ones(len(Fcal))])
            pf = lambda d: iso.predict(clf.predict_proba(d[feats].values)[:, 1])
            pi, pfa = pf(H), pf(Fte)
            r = dict(auc=float(roc_auc_score(np.r_[np.zeros(len(pi)), np.ones(len(pfa))], np.r_[pi, pfa])))
            nt = H.d_true.between(*NEAR).values
            r["auc true near-fold intrinsic vs fast"] = float(roc_auc_score(np.r_[np.zeros(nt.sum()), np.ones(len(pfa))], np.r_[pi[nt], pfa]))
            r["p_fast simulated deep foci (true delta < -3)"] = float(pi[H.d_true.values < -3].mean())
            r["p_fast simulated true near-fold intrinsic"] = float(pi[nt].mean())
            r["p_fast simulated fast modulation"] = float(pfa.mean())
            for t, (da, tw) in emp.items():
                a, b = pf(da), pf(tw)
                nf = np.asarray(fold_delta(da.th_mu_q50, da.th_c_q50)) > -1
                r[t] = dict(n_data=int(len(da)), p_fast_twin=float(b.mean()), p_fast_data=float(a.mean()),
                            p_fast_data_near_fold=float(a[nf].mean()) if nf.any() else None)
            # domain classifier: simulated deep (linear) foci against empirical twins (subsampled to equal size)
            T = pd.concat([tw for _, tw in emp.values()])[feats]
            deep = pool[fold_delta(pool.th_mu, pool.th_c) < -2.5]
            n = min(len(T), 6000, len(deep))
            S = deep[feats].sample(n, random_state=1); T = T.sample(n, random_state=1)
            X = np.vstack([S.values, T.values]); y = np.r_[np.zeros(n), np.ones(n)]
            p = cross_val_predict(HistGradientBoostingClassifier(max_iter=300, random_state=0), X, y, cv=5, method="predict_proba")[:, 1]
            r["domain auc: simulations vs empirical twins"] = float(roc_auc_score(y, p))
            R[sname] = r
            print(cfg, sname, json.dumps(r, indent=1), flush=True)
        rep[cfg] = R
    json.dump(rep, open("results/mech_domain.json", "w"), indent=1)


if __name__ == "__main__":
    main()

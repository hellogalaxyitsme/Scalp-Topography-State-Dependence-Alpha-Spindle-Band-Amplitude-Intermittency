"""Near-fold comparison of EEG and simulations with one selection interval for both (estimated delta between -1.5
and 1) and participant-grouped cross-validation of the domain classifiers: all channels and pieces of a participant
(both nights of a Sleep-EDF participant) stay in the same fold, simulations are independent. Classifier scores of
the main classifier of mech_twin_broad.py (isotonic calibration on simulations) for held-out simulations of every
class selected by estimated delta (deep foci: true delta < -3) and for near-fold EEG; negative controls (twin minus
an independent twin) against simulated deep foci with grouped folds.
usage: python scripts/mech_nearfold.py  -> results/mech_nearfold.json, results/mech_nearfold_points.csv"""
import json, joblib, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sbi_fit import predict
from ampint.nf import fold_delta
from mech_twin import load, differenced, DIFF, NEAR, TABLES
from mech_twin_broad import twin2, TWIN2, HEALTHY_SUBS

DCOLS = ["d_" + f for f in DIFF]


def unit_of(t, D):
    return D["sub"] if t == "tdbrain" else (D.rec.str[:5] if t.startswith("sleep") else D.rec)


def domain(A, B, gb, grouped, cols=DCOLS, n=6000):
    """Cross-validated AUC separating A (simulations, independent) from B (EEG or controls, groups gb)."""
    ia = np.random.default_rng(1).permutation(len(A))[:min(n, len(A), len(B))]
    ib = np.random.default_rng(1).permutation(len(B))[:len(ia)]
    X = np.vstack([A[cols].values[ia], B[cols].values[ib]])
    y = np.r_[np.zeros(len(ia)), np.ones(len(ib))]
    g = np.r_[-1 - np.arange(len(ia)), pd.factorize(np.asarray(gb)[ib])[0]]
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=0) if grouped else StratifiedKFold(5, shuffle=True, random_state=0)
    p = cross_val_predict(HistGradientBoostingClassifier(max_iter=300, random_state=0), X, y, groups=g if grouped else None,
                          cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def main():
    out, pts = {}, []
    for cfg, tables in TABLES.items():
        A = joblib.load(f"results/mechtwb_{cfg}.joblib")
        score = lambda d: A["iso"].predict(A["clf"].predict_proba(d[A["cols"]].values)[:, 1])
        reg = joblib.load(f"results/sbi_{cfg}_full.joblib")
        C = {}
        for k in ("intrinsic", "mult", "fast"):
            d = load(cfg, k)
            d["split"] = np.random.default_rng(0 if k != "mult" else 7).choice(["tr", "cal", "te"], len(d), p=[0.8, 0.1, 0.1])
            pr = predict(reg, d)
            d["d_hat"] = fold_delta(pr.th_mu_q50.values, pr.th_c_q50.values); d["d_true"] = fold_delta(d.th_mu.values, d.th_c.values)
            C[k] = d
        near_sim = {k: d[d.d_hat.between(*NEAR)] for k, d in C.items()}
        deep = C["intrinsic"][C["intrinsic"].d_true < -2.5]
        r = {"simulations near fold, median estimated delta": {k: float(d.d_hat.median()) for k, d in near_sim.items()}}
        held = {"deep focus, additive": C["intrinsic"][(C["intrinsic"].split == "te") & (C["intrinsic"].d_true < -3)]}
        held.update({f"near-fold estimate, {lab}": near_sim[k][near_sim[k].split == "te"]
                     for k, lab in (("intrinsic", "additive"), ("mult", "multiplicative"), ("fast", "fast modulation"))})
        r["held-out simulations"] = {}
        for lab, d in held.items():
            s = score(d)
            r["held-out simulations"][lab] = dict(n=int(len(d)), mean_score=float(s.mean()), frac_gt_0_5=float((s > 0.5).mean()))
            pts += [dict(cfg=cfg, category=lab, p=v) for v in s]
        for t, keys in tables.items():
            P = pd.read_csv(f"results/post_{t}.csv")
            if t == "tdbrain":
                P = P[P["sub"].isin(HEALTHY_SUBS) & (P.ses == "ses-1")]
            P["delta"] = fold_delta(P.th_mu_q50.values, P.th_c_q50.values)
            D = differenced(P[P.kind == "data"], P[P.kind == "iaaft0"], keys)
            D["p_fast"] = score(D)
            old, new = D.delta > -1, D.delta.between(*NEAR)
            e = {"previous selection (delta > -1)": dict(n=int(old.sum()), mean_score=float(D.p_fast[old].mean()),
                                                         frac_gt_0_5=float((D.p_fast[old] > 0.5).mean()),
                                                         frac_with_delta_above_1=float((D.delta[old] > 1).mean())),
                 "common selection (-1.5 < delta < 1)": dict(n=int(new.sum()), mean_score=float(D.p_fast[new].mean()),
                                                             frac_gt_0_5=float((D.p_fast[new] > 0.5).mean()),
                                                             median_delta=float(D.delta[new].median()))}
            N = D[new]; gN = unit_of(t, N)
            e["domain auc, common selection"] = {k: dict(random_folds=domain(near_sim[k], N, gN, False), participant_folds=domain(near_sim[k], N, gN, True))
                                                 for k in ("intrinsic", "mult", "fast")}
            ctrl = None
            if (P.kind == "iaaft1").any():
                ctrl = differenced(P[P.kind == "iaaft1"], P[P.kind == "iaaft0"], keys)
            elif t in TWIN2:
                T2 = twin2(t)
                if len(T2):
                    ctrl = differenced(T2, P[P.kind == "iaaft0"], keys)
            if ctrl is not None and len(ctrl):
                gc = unit_of(t, ctrl)
                e["negative control vs simulated deep foci"] = dict(random_folds=domain(deep, ctrl, gc, False), participant_folds=domain(deep, ctrl, gc, True))
            pts += [dict(cfg=cfg, category=f"EEG near fold, {t}", p=v) for v in N.p_fast.values]
            r[t] = e
            print(cfg, t, json.dumps(e), flush=True)
        out[cfg] = r
    json.dump(out, open("results/mech_nearfold.json", "w"), indent=1)
    pd.DataFrame(pts).to_csv("results/mech_nearfold_points.csv", index=False)


if __name__ == "__main__":
    main()

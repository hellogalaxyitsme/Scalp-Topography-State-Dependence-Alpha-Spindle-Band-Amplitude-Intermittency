"""Which summary statistics drive the mechanism classifier and the separation of EEG from simulations. Statistics are
grouped into families (closed-form coordinates, amplitude shape, log-power quantiles, log-power histogram, short-lag
and long-lag power autocorrelation, context). (1) Main classifier of mech_twin_broad.py: drop in held-out AUC (intrinsic
versus fast modulation) when the twin differences of one family are permuted jointly (5 repeats). (2) Domain
classifiers (near-fold EEG versus near-fold simulations): cross-validated AUC using one family alone and using all
families except one.
usage: python scripts/mech_drivers.py  -> results/mech_drivers.json"""
import json, joblib, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sbi_fit import predict
from ampint.nf import fold_delta
from ampint.summ import DLAGS
from mech_twin import load, differenced, domain_auc, DIFF, CTX, NEAR

R0 = pd.read_csv("results/nf_tdbrain_reclevel.csv")
HEALTHY_SUBS = set(R0.loc[R0.group == "HEALTHY", "sub"])
EEG = {"alpha": ("tdbrain", ["sub", "ses", "task", "ch"]), "sigma250": ("anphy_sigma", ["rec", "ch", "stage", "chunk"])}


def family(f):
    if f in ("mu_d", "c_d", "mu_n", "c_n"):
        return "closed-form coordinates"
    if f in ("cv", "kurt"):
        return "amplitude shape"
    if f.startswith("lq"):
        return "log-power quantiles"
    if f.startswith("h") and f[1:].isdigit():
        return "log-power histogram"
    lag = float(f[3:]) if f.startswith("acf") else DLAGS[int(f[1:])]
    return "autocorrelation, lags up to 0.3 s" if lag <= 0.3 else "autocorrelation, lags above 0.3 s"


def main():
    groups = {}
    for f in DIFF:
        groups.setdefault(family(f), []).append("d_" + f)
    groups["context"] = list(CTX)
    out = dict(families={k: len(v) for k, v in groups.items()})
    rng = np.random.default_rng(0)
    for cfg, (table, keys) in EEG.items():
        A = joblib.load(f"results/mechtwb_{cfg}.joblib")
        clf, iso, cols = A["clf"], A["iso"], A["cols"]
        C = {k: load(cfg, k) for k in ("intrinsic", "mult", "fast")}
        for k, d in C.items():
            d["split"] = np.random.default_rng(0 if k != "mult" else 7).choice(["tr", "cal", "te"], len(d), p=[0.8, 0.1, 0.1])
        te = pd.concat([C["intrinsic"][C["intrinsic"].split == "te"], C["mult"][C["mult"].split == "te"], C["fast"][C["fast"].split == "te"]])
        y = np.r_[np.zeros((C["intrinsic"].split == "te").sum() + (C["mult"].split == "te").sum()), np.ones((C["fast"].split == "te").sum())]
        X = te[cols].values
        base = float(roc_auc_score(y, clf.predict_proba(X)[:, 1]))
        imp = {}
        for g, gc in groups.items():
            idx = [cols.index(c) for c in gc]
            drops = []
            for _ in range(5):
                Xp = X.copy(); Xp[:, idx] = X[rng.permutation(len(X))][:, idx]
                drops.append(base - roc_auc_score(y, clf.predict_proba(Xp)[:, 1]))
            imp[g] = float(np.mean(drops))
        r = {"classifier auc": base, "auc drop when family permuted": imp}
        reg = joblib.load(f"results/sbi_{cfg}_full.joblib")
        P = pd.read_csv(f"results/post_{table}.csv")
        if table == "tdbrain":
            P = P[P["sub"].isin(HEALTHY_SUBS) & (P.ses == "ses-1")]
        P["delta"] = fold_delta(P.th_mu_q50.values, P.th_c_q50.values)
        D = differenced(P[P.kind == "data"], P[P.kind == "iaaft0"], keys)
        D = D[D.delta > -1]
        dom = {}
        for k in ("intrinsic", "fast"):
            S = C[k]; pr = predict(reg, S)
            dh = fold_delta(pr.th_mu_q50.values, pr.th_c_q50.values)
            S = S[(dh >= NEAR[0]) & (dh <= NEAR[1])]
            allc = [c for c in cols if c not in CTX]
            e = {"all twin differences": domain_auc(S, D, allc)}
            e["one family alone"] = {g: domain_auc(S, D, gc) for g, gc in groups.items() if g != "context"}
            e["all but one family"] = {g: domain_auc(S, D, [c for c in allc if c not in gc]) for g, gc in groups.items() if g != "context"}
            dom[k] = e
        r["domain auc, near-fold EEG versus near-fold simulations"] = dom
        out[cfg] = r
        print(cfg, json.dumps(r, indent=1), flush=True)
    json.dump(out, open("results/mech_drivers.json", "w"), indent=1)


if __name__ == "__main__":
    main()

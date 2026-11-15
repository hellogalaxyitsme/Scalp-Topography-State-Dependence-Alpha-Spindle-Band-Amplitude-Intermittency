"""Amortised posterior for stochastic-Bautin coordinates from summary statistics.

* Quantile gradient boosting (ABC-random-forest spirit; Raynal et al. 2019) for (mu_t, c_t, log10 t0).
* Split-conformal calibration of the 50% and 90% intervals (CQR; Romano et al. 2019) on a calibration split.
* Regime classifier (focus / bistable / oscillatory of the TRUE parameters) -> posterior regime probabilities.
* Amortised adequacy test: quantile model of the long-lag power-ACF tail (2-10 s) expected under the 1-D
  normal form, from short-lag features only; data above the calibrated 95% quantile => hidden slow variable.
usage: python scripts/sbi_fit.py <config> [tag]
"""
import sys, os, glob, json, numpy as np, pandas as pd, joblib
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from ampint.summ import FEATURES, DLAGS
from ampint.nf import regime

QUANT = [0.05, 0.25, 0.5, 0.75, 0.95]
TARGETS = ["th_mu", "th_c", "log_t0"]
NUIS = ["th_snr_db", "th_chi"]
TAIL_FEATURES = [f for f in FEATURES if not (f.startswith("acf") and float(f[3:]) > 1.2)
                 and not (f.startswith("a") and f[1:].isdigit() and DLAGS[int(f[1:])] > 1.2)]


def load_bank(cfg, kind="bank"):
    fs = sorted(glob.glob(f"results/{kind}_{cfg}/chunk_*.npz"))
    lst = f"config/{kind}_{cfg}_chunks.txt"   # optional list of bank chunks to train on
    if os.path.exists(lst):
        keep = set(open(lst).read().split())
        fs = [f for f in fs if os.path.basename(f) in keep]
    df = pd.concat([pd.DataFrame(dict(np.load(f, allow_pickle=True))) for f in fs], ignore_index=True)
    df["log_t0"] = np.log10(df["th_t0"])
    df["regime"] = [regime(m, c) for m, c in zip(df.th_mu, df.th_c)]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + TARGETS + ["tail"])
    return df


def hgb_q(q, it=800):
    return HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=it, learning_rate=0.05,
                                         max_leaf_nodes=63, min_samples_leaf=40, l2_regularization=1.0, random_state=0)


def fit_models(tr, cal):
    X, Xc = tr[FEATURES].values, cal[FEATURES].values
    M = dict(features=FEATURES, tail_features=TAIL_FEATURES)
    for t in TARGETS + NUIS:
        qs = QUANT if t in TARGETS else [0.5]
        M[t] = {q: hgb_q(q).fit(X, tr[t].values) for q in qs}
        if t in TARGETS:  # CQR conformity offsets for 50% and 90% intervals
            pc = {q: M[t][q].predict(Xc) for q in QUANT}
            y = cal[t].values
            for lo, hi, a in [(0.25, 0.75, 0.5), (0.05, 0.95, 0.1)]:
                s = np.maximum(pc[lo] - y, y - pc[hi])
                k = int(np.ceil((len(y) + 1) * (1 - a))) - 1
                M[t][f"cqr_{lo}_{hi}"] = float(np.sort(s)[min(k, len(s) - 1)])
    M["regime"] = HistGradientBoostingClassifier(max_iter=600, learning_rate=0.05, max_leaf_nodes=63,
                                                 min_samples_leaf=40, random_state=0).fit(X, tr["regime"].values)
    Xt, Xtc = tr[TAIL_FEATURES].values, cal[TAIL_FEATURES].values
    M["tail"] = {q: hgb_q(q).fit(Xt, tr["tail"].values) for q in [0.5, 0.95]}
    s = cal["tail"].values - M["tail"][0.95].predict(Xtc)          # conformal one-sided 95% bound
    M["tail_cqr95"] = float(np.sort(s)[int(np.ceil((len(s) + 1) * 0.95)) - 1])
    return M


def predict(M, df):
    X = df[M["features"]].values
    out = pd.DataFrame(index=df.index)
    for t in TARGETS:
        qs = np.sort(np.stack([M[t][q].predict(X) for q in QUANT], 1), 1)
        for i, q in enumerate(QUANT):
            out[f"{t}_q{int(q * 100)}"] = qs[:, i]
        out[f"{t}_lo50"] = qs[:, 1] - M[t]["cqr_0.25_0.75"]; out[f"{t}_hi50"] = qs[:, 3] + M[t]["cqr_0.25_0.75"]
        out[f"{t}_lo90"] = qs[:, 0] - M[t]["cqr_0.05_0.95"]; out[f"{t}_hi90"] = qs[:, 4] + M[t]["cqr_0.05_0.95"]
    for t in NUIS:
        out[t + "_hat"] = M[t][0.5].predict(X)
    pr = M["regime"].predict_proba(X)
    for i, c in enumerate(M["regime"].classes_):
        out[f"p_{c}"] = pr[:, i]
    Xt = df[M["tail_features"]].values
    out["tail_pred50"] = M["tail"][0.5].predict(Xt)
    out["tail_bound95"] = M["tail"][0.95].predict(Xt) + M["tail_cqr95"]
    out["slow_excess"] = df["tail"].values - out["tail_pred50"].values
    out["slow_flag"] = (df["tail"].values > out["tail_bound95"].values).astype(int)
    return out


def report(te, pr):
    rep = {}
    for t in TARGETS:
        y = te[t].values; m = pr[f"{t}_q50"].values
        rep[t] = dict(r2=float(1 - np.mean((y - m) ** 2) / np.var(y)), mae=float(np.mean(np.abs(y - m))),
                      cov90=float(np.mean((y >= pr[f"{t}_lo90"]) & (y <= pr[f"{t}_hi90"]))),
                      cov50=float(np.mean((y >= pr[f"{t}_lo50"]) & (y <= pr[f"{t}_hi50"]))),
                      width90=float(np.mean(pr[f"{t}_hi90"] - pr[f"{t}_lo90"])))
    cl = [c[2:] for c in pr.columns if c.startswith("p_")]
    pred_reg = np.array(cl)[np.argmax(pr[[f"p_{c}" for c in cl]].values, 1)]
    rep["regime_acc"] = float(np.mean(pred_reg == te["regime"].values))
    rep["regime_confusion"] = {str(k): v for k, v in pd.crosstab(te["regime"].values, pred_reg).to_dict().items()}
    big = np.abs(te.th_c.values) > 1
    rep["c_sign_acc_|c|>1"] = float(np.mean((np.sign(pr["th_c_q50"].values) == np.sign(te.th_c.values))[big]))
    rep["tail_false_flag_rate"] = float(pr["slow_flag"].mean())
    return rep


if __name__ == "__main__":
    cfg = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else "full"
    df = load_bank(cfg)
    rng = np.random.default_rng(0)
    u = rng.random(len(df))
    tr, cal, te = df[u < 0.8], df[(u >= 0.8) & (u < 0.9)], df[u >= 0.9]
    M = fit_models(tr, cal)
    pr = predict(M, te)
    rep = dict(n_train=int(len(tr)), n_cal=int(len(cal)), n_test=int(len(te)), **report(te, pr))
    # near-Bautin region
    near = (te.th_mu > -3).values
    rep["near_bautin"] = report(te[near], pr[near])
    # adequacy test power on modulated simulations, if available
    if glob.glob(f"results/modbank_{cfg}/chunk_*.npz"):
        mb = load_bank(cfg, "modbank")
        pm = predict(M, mb)
        rep["modulated"] = dict(n=int(len(mb)), flag_rate=float(pm.slow_flag.mean()),
                                flag_rate_by_sd={str(k): float(v) for k, v in pm.slow_flag.groupby(pd.cut(mb.mod_sd, [0, .5, 1, 1.5, 2.6])).mean().items()},
                                c_bias=float(np.median(pm.th_c_q50 - mb.th_c)), mu_bias=float(np.median(pm.th_mu_q50 - mb.th_mu)))
        pd.concat([mb.reset_index(drop=True), pm.reset_index(drop=True)], axis=1).to_csv(f"results/sbi_{cfg}_{tag}_modulated.csv", index=False)
    print(json.dumps(rep, indent=1, default=str))
    joblib.dump(M, f"results/sbi_{cfg}_{tag}.joblib")
    pd.concat([te.reset_index(drop=True), pr.reset_index(drop=True)], axis=1).to_csv(f"results/sbi_{cfg}_{tag}_heldout.csv", index=False)
    json.dump(rep, open(f"results/sbi_{cfg}_{tag}_report.json", "w"), indent=1, default=str)

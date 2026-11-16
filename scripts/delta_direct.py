"""Direct estimator of the fold distance: quantile gradient-boosted trees trained on the true delta of the simulation
bank (same summaries, splits and conformal calibration as sbi_fit.py), compared with the plug-in estimate from the
posterior medians of mu_t and c_t on the held-out simulations (whole prior and healthy-alpha region), and applied to
the empirical summaries: eye opening (TDBRAIN healthy adults, 26-channel map and O1/Oz/O2/Pz mean) and N2 minus
wakefulness in the spindle band (ANPHY-Sleep scalp channels, Sleep-EDF Fpz-Cz). Effect sizes average pieces and
channels within participant as descriptor_benchmark.py; cluster maps use the aggregation and permutation seeds of
topo_tdbrain.py and analyze_anphy.py (ANPHY-Sleep pieces summarized by their median per night, channel and stage).
usage: python scripts/delta_direct.py <n_perm>  -> results/delta_direct.json"""
import sys, json, numpy as np, pandas as pd, mne
from scipy.stats import spearmanr
from sbi_fit import load_bank, hgb_q, QUANT
from ampint.summ import FEATURES
from ampint.nf import fold_delta
from ampint.io_tdbrain import load_meta, EEG26
from ampint.io_anphy import positions
from ampint.stats import channel_adjacency, cluster_test

NPERM = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
BOX = ((-7.0, -3.0), (1.5, 5.5))
POST4 = {"O1", "OZ", "O2", "PZ"}


def train(cfg):
    df = load_bank(cfg)
    df["delta"] = fold_delta(df.th_mu.values, df.th_c.values)
    u = np.random.default_rng(0).random(len(df))
    tr, cal, te = df[u < 0.8], df[(u >= 0.8) & (u < 0.9)], df[u >= 0.9]
    M = {q: hgb_q(q).fit(tr[FEATURES].values, tr.delta.values) for q in QUANT}
    pc = {q: M[q].predict(cal[FEATURES].values) for q in QUANT}
    s = np.maximum(pc[0.05] - cal.delta.values, cal.delta.values - pc[0.95])
    off = float(np.sort(s)[int(np.ceil((len(s) + 1) * 0.9)) - 1])
    return M, off, te


def predict_delta(M, X):
    return np.sort(np.stack([M[q].predict(X) for q in QUANT], 1), 1)


def metrics(y, x, lo=None, hi=None):
    e = x - y
    r = dict(n=int(len(y)), r2=float(1 - np.mean(e ** 2) / np.var(y)), rs=float(spearmanr(x, y)[0]),
             bias_median=float(np.median(e)), mae=float(np.mean(np.abs(e))))
    if lo is not None:
        r["coverage90"] = float(np.mean((y >= lo) & (y <= hi)))
    return r


def dz(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(x.mean() / x.std(ddof=1))


def main():
    out = {}
    models = {}
    for cfg in ("alpha", "sigma250"):
        M, off, te = train(cfg)
        models[cfg] = M
        H = pd.read_csv(f"results/sbi_{cfg}_full_heldout.csv")
        H["delta"] = fold_delta(H.th_mu.values, H.th_c.values)
        H["plug"] = fold_delta(H.th_mu_q50.values, H.th_c_q50.values)
        q = predict_delta(M, H[FEATURES].values)
        H["direct"], lo, hi = q[:, 2], q[:, 0] - off, q[:, 4] + off
        box = (H.th_mu.between(*BOX[0]) & H.th_c.between(*BOX[1])).values
        out[f"recovery {cfg}"] = {
            region: {"plug-in": metrics(H.delta.values[m], H.plug.values[m]),
                     "direct": metrics(H.delta.values[m], H.direct.values[m], lo[m], hi[m]),
                     "rs plug-in vs direct": float(spearmanr(H.plug.values[m], H.direct.values[m])[0])}
            for region, m in (("whole prior", np.ones(len(H), bool)), ("healthy region", box))}
    # eye opening, TDBRAIN healthy adults, first session
    P = pd.read_csv("results/post_tdbrain.csv", usecols=["sub", "ses", "task", "ch", "kind", "th_mu_q50", "th_c_q50"] + FEATURES)
    P = P[(P.kind == "data") & (P.ses == "ses-1")].replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)
    meta = load_meta(); m1 = meta[meta.sessID == 1].drop_duplicates("sub")[["sub", "indication", "age", "gender"]]
    P = P.merge(m1, on="sub"); P = P[(P.indication == "HEALTHY") & (P.age >= 18) & P.gender.notna()].copy()
    P["plug"] = fold_delta(P.th_mu_q50.values, P.th_c_q50.values)
    P["direct"] = predict_delta(models["alpha"], P[FEATURES].values)[:, 2]
    mont = mne.channels.make_standard_montage("standard_1020").get_positions()["ch_pos"]
    pos = {k.upper(): v for k, v in mont.items()}
    chans = list(EEG26); adj = channel_adjacency([c.upper() for c in chans], pos)
    eo = {"rs plug-in vs direct (channel recordings)": float(spearmanr(P.plug, P.direct)[0])}
    for est in ("plug", "direct"):
        q4 = P[P.ch.str.upper().isin(POST4)].groupby(["sub", "task"])[est].mean().unstack().dropna()
        w = P.pivot_table(index=["sub", "ch"], columns="task", values=est).dropna()
        D = (w["restEO"] - w["restEC"]).unstack("ch").reindex(columns=chans)
        t, cl = cluster_test(D.values, adj, n_perm=NPERM, seed=1)
        eo[est] = dict(n=int(len(q4)), dz_post4=dz(q4.restEO - q4.restEC),
                       clusters=[dict(sign=c["sign"], channels=[chans[i] for i in c["channels"]], p=c["p"]) for c in cl if c["p"] < 0.05])
    out["EO-EC (TDBRAIN healthy)"] = eo
    # N2 - wake, spindle band
    sl = {}
    for t, chf in (("anphy_sigma", None), ("sleep_sigma", "EEG Fpz-Cz")):
        S = pd.read_csv(f"results/post_{t}.csv", usecols=["rec", "ch", "stage", "kind", "th_mu_q50", "th_c_q50"] + FEATURES)
        S = S[(S.kind == "data") & S.stage.isin(["W", "N2"])].replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)
        S = S[(S.ch == chf) if chf else ~S.ch.str.contains("-")].copy()
        if chf:
            S["rec"] = S.rec.str[:5]
        S["plug"] = fold_delta(S.th_mu_q50.values, S.th_c_q50.values)
        S["direct"] = predict_delta(models["sigma250"], S[FEATURES].values)[:, 2]
        r = {"rs plug-in vs direct (pieces)": float(spearmanr(S.plug, S.direct)[0])}
        for est in ("plug", "direct"):
            G = S.groupby(["rec", "stage"])[est].mean().unstack().dropna()
            e = dict(n=int(len(G)), dz=dz(G.N2 - G.W), median_W=float(G.W.median()), median_N2=float(G.N2.median()))
            if chf is None:
                ps = positions(); scalp = [c for c in sorted(S.ch.unique()) if c in ps]
                adj2 = channel_adjacency(scalp, ps)
                w = S[S.ch.isin(scalp)].groupby(["rec", "ch", "stage"])[est].median().unstack()
                D = (w["N2"] - w["W"]).unstack("ch").reindex(columns=scalp)
                _, cl = cluster_test(D.values, adj2, n_perm=NPERM, seed=3)
                e["clusters"] = [dict(sign=c["sign"], n_channels=len(c["channels"]), p=c["p"]) for c in cl if c["p"] < 0.05]
            r[est] = e
        sl[t] = r
    out["N2-W"] = sl
    json.dump(out, open("results/delta_direct.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

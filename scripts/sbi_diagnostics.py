"""Additional inference diagnostics.
(1) Held-out recovery of the signed fold distance delta = mu + c^2/4 (c > 0; delta = mu otherwise), with
    split-conformal intervals calibrated on the calibration split.
(2) Calibration of the regime probabilities (Brier score, expected calibration error, reliability table).
(3) Coverage of empirical summary vectors by the simulation bank (k-nearest-neighbour distances in the
    standardised summary space, and empirical values outside the simulated range of key statistics).
usage: python scripts/sbi_diagnostics.py  -> results/sbi_diagnostics.json"""
import json, joblib, numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
from sbi_fit import load_bank, predict
from ampint.summ import FEATURES
from sbi_bank import CONFIGS

HEALTHY = dict(mu=(-7.0, -3.0), c=(1.5, 5.5))


def delta(mu, c):
    return np.where(c > 0, mu + c * c / 4, mu)


def split(cfg):
    df = load_bank(cfg)
    u = np.random.default_rng(0).random(len(df))
    return df[u < 0.8], df[(u >= 0.8) & (u < 0.9)], df[u >= 0.9]


def recovery(y, m):
    return dict(n=int(len(y)), r2=float(1 - np.mean((y - m) ** 2) / np.var(y)), mae=float(np.mean(np.abs(y - m))),
                bias=float(np.median(m - y)), spearman=float(pd.Series(y).corr(pd.Series(m), method="spearman")))


def delta_block(cal, pc, te, pt):
    d_cal = delta(cal.th_mu.values, cal.th_c.values); dh_cal = delta(pc.th_mu_q50.values, pc.th_c_q50.values)
    s = np.sort(np.abs(dh_cal - d_cal)); n = len(s)
    q90 = float(s[min(n - 1, int(np.ceil((n + 1) * 0.9)) - 1)]); q50 = float(s[min(n - 1, int(np.ceil((n + 1) * 0.5)) - 1)])
    d = delta(te.th_mu.values, te.th_c.values); dh = delta(pt.th_mu_q50.values, pt.th_c_q50.values)
    regions = {"all": np.ones(len(te), bool),
               "healthy operating region": te.th_mu.between(*HEALTHY["mu"]).values & te.th_c.between(*HEALTHY["c"]).values,
               "near fold (|delta|<2, c>0)": (np.abs(d) < 2) & (te.th_c.values > 0)}
    out = dict(halfwidth90=q90, halfwidth50=q50)
    for name, m in regions.items():
        r = dict(delta=recovery(d[m], dh[m]))
        r["delta"]["cov90"] = float(np.mean(np.abs(dh[m] - d[m]) <= q90))
        r["delta"]["cov50"] = float(np.mean(np.abs(dh[m] - d[m]) <= q50))
        for t in ["th_mu", "th_c"]:
            r[t] = recovery(te[t].values[m], pt[f"{t}_q50"].values[m])
            r[t]["cov90"] = float(np.mean((te[t].values[m] >= pt[f"{t}_lo90"].values[m]) & (te[t].values[m] <= pt[f"{t}_hi90"].values[m])))
        out[name] = r
    return out


def calibration(te, pt, nbins=10):
    cl = ["focus", "bistable", "oscillatory"]
    P = pt[[f"p_{c}" for c in cl]].values; Y = np.stack([(te.regime.values == c).astype(float) for c in cl], 1)
    out = dict(brier=float(np.mean(np.sum((P - Y) ** 2, 1))),
               brier_climatology=float(np.mean(np.sum((Y.mean(0) - Y) ** 2, 1))))
    edges = np.linspace(0, 1, nbins + 1)
    for i, c in enumerate(cl):
        p, y = P[:, i], Y[:, i]
        b = np.clip(np.digitize(p, edges) - 1, 0, nbins - 1)
        rows = [dict(lo=float(edges[k]), hi=float(edges[k + 1]), n=int((b == k).sum()),
                     mean_p=float(p[b == k].mean()) if (b == k).any() else None,
                     freq=float(y[b == k].mean()) if (b == k).any() else None) for k in range(nbins)]
        ece = float(sum(r["n"] * abs(r["mean_p"] - r["freq"]) for r in rows if r["n"]) / len(p))
        out[c] = dict(ece=ece, brier=float(np.mean((p - y) ** 2)), reliability=rows)
    return out


def coverage(cfg, tr, te, emp):
    """emp: dict name -> DataFrame of empirical summary rows."""
    mu, sd = tr[FEATURES].mean(), tr[FEATURES].std().replace(0, 1)
    Z = lambda d: ((d[FEATURES] - mu) / sd).values
    ref = tr.sample(min(40000, len(tr)), random_state=0)
    nn = NearestNeighbors(n_neighbors=10).fit(Z(ref))
    d_te = nn.kneighbors(Z(te))[0].mean(1)
    p95, p99 = np.percentile(d_te, [95, 99])
    simrange = {k: (float(np.percentile(tr[k], 0.5)), float(np.percentile(tr[k], 99.5))) for k in ["rho", "T", "cv", "kurt", "peak"]}
    out = dict(sim_knn_p95=float(p95), sim_knn_p99=float(p99), sim_range_0_5_99_5=simrange, f0_prior=list(CONFIGS[cfg]["f0"]))
    for name, e in emp.items():
        e = e.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)
        d = nn.kneighbors(Z(e))[0].mean(1)
        r = dict(n=int(len(e)), frac_knn_above_sim_p95=float(np.mean(d > p95)), frac_knn_above_sim_p99=float(np.mean(d > p99)),
                 median_knn_percentile=float(np.median(np.searchsorted(np.sort(d_te), d) / len(d_te) * 100)))
        r["frac_outside_sim_range"] = {k: float(np.mean((e[k] < lo) | (e[k] > hi))) for k, (lo, hi) in simrange.items()}
        r["frac_peak_outside_f0_prior"] = float(np.mean((e.peak < CONFIGS[cfg]["f0"][0]) | (e.peak > CONFIGS[cfg]["f0"][1])))
        r["frac_T_below_60s"] = float(np.mean(e["T"] < 60))
        out[name] = r
    return out


def main():
    out = {}
    POST = ["O1", "Oz", "O2", "Pz", "P3", "P4"]
    for cfg in ["alpha", "sigma250"]:
        tr, cal, te = split(cfg)
        M = joblib.load(f"results/sbi_{cfg}_full.joblib")
        pc, pt = predict(M, cal).reset_index(drop=True), predict(M, te).reset_index(drop=True)
        cal, te = cal.reset_index(drop=True), te.reset_index(drop=True)
        out[cfg] = dict(delta=delta_block(cal, pc, te, pt), calibration=calibration(te, pt))
        cols = FEATURES + ["peak"]
        if cfg == "alpha":
            tb = pd.read_csv("results/post_tdbrain.csv", usecols=cols + ["kind", "ch"])
            mz = pd.read_csv("results/post_mumtaz.csv", usecols=cols + ["ch"])
            dd = pd.read_csv("results/post_ds003478.csv", usecols=cols + ["ch"])
            sa = pd.read_csv("results/post_sleep_alpha.csv", usecols=cols + ["kind"])
            emp = {"TDBRAIN (posterior channels)": tb[(tb.kind == "data") & tb.ch.isin(POST)],
                   "TDBRAIN (all channels)": tb[tb.kind == "data"],
                   "Mumtaz": mz[mz.ch.isin(["O1", "O2", "P3", "P4", "Pz"])],
                   "ds003478": dd[dd.ch.isin(["O1", "OZ", "O2", "PZ", "P3", "P4", "POZ", "PO3", "PO4"])],
                   "Sleep-EDF alpha band": sa[sa.kind == "data"]}
        else:
            ss = pd.read_csv("results/post_sleep_sigma.csv", usecols=cols + ["kind"])
            emp = {"Sleep-EDF sigma band": ss[ss.kind == "data"]}
        out[cfg]["coverage"] = coverage(cfg, tr, te, emp)
        print(cfg, json.dumps({k: v for k, v in out[cfg].items() if k != "calibration"}, indent=1, default=str)[:6000], flush=True)
        print(cfg, "calibration", json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "reliability"})
                                              for k, v in out[cfg]["calibration"].items()}, indent=1), flush=True)
    json.dump(out, open("results/sbi_diagnostics.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()

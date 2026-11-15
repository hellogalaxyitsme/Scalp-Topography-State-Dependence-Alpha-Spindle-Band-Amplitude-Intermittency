"""Peak selection: how often the selected peak lies outside the simulated oscillator-frequency prior or barely
exceeds the aperiodic fit, and whether the central state comparisons hold in the subset with a credible peak.
Credible peak: log10 peak height above the aperiodic fit >= 0.1 (26% above) and frequency inside the simulated
prior (alpha 8-12.5 Hz, spindle band 11.5-14.5 Hz). The EEG-twin comparison uses the data peak for both.
usage: python scripts/peak_sensitivity.py [n_perm]"""
import sys, json, numpy as np, pandas as pd, mne
NPERM = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
sys.argv = [sys.argv[0], str(NPERM)]
from ampint.io_tdbrain import load_meta, EEG26
from ampint.io_anphy import positions
from ampint.nf import fold_delta
from ampint.stats import channel_adjacency, cluster_test
import analyze_anphy as AA

PRIOR_F = {"alpha": (8.0, 12.5), "sigma": (11.5, 14.5)}
HMIN = 0.1
POST4 = {"O1", "OZ", "O2", "PZ"}


def flags(d, band):
    lo, hi = PRIOR_F[band]
    d["in_prior"] = d.peak.between(lo, hi)
    d["high"] = d.peak_height >= HMIN
    d["credible"] = d.in_prior & d.high
    return d


def frac(d, by):
    g = d.groupby(by)
    return pd.DataFrame(dict(n=g.size(), outside_prior=1 - g.in_prior.mean(), low_peak=1 - g.high.mean(),
                             not_credible=1 - g.credible.mean())).round(3)


def main():
    out = {}
    # TDBRAIN alpha (healthy adults, first session)
    p = pd.read_csv("results/post_tdbrain.csv", usecols=["sub", "ses", "task", "ch", "kind", "peak", "peak_height", "th_mu_q50", "th_c_q50"])
    p = flags(p[p.ses == "ses-1"].copy(), "alpha"); p["delta"] = fold_delta(p.th_mu_q50, p.th_c_q50)
    meta = load_meta(); m1 = meta[meta.sessID == 1].drop_duplicates("sub")[["sub", "indication", "age", "gender"]]
    p = p.merge(m1, on="sub"); p = p[(p.indication == "HEALTHY") & (p.age >= 18) & p.gender.notna()]
    d = p[p.kind == "data"]
    out["TDBRAIN fractions"] = {"all channels": frac(d, "task").to_dict(orient="index"),
                                "O1/Oz/O2/Pz": frac(d[d.ch.str.upper().isin(POST4)], "task").to_dict(orient="index")}
    out["TDBRAIN selected peak frequency (median, IQR)"] = {t: d[d.task == t].peak.quantile([.25, .5, .75]).round(2).tolist() for t in ("restEC", "restEO")}
    res = {}
    for tag, dd in (("all", d), ("credible", d[d.credible])):
        q = dd[dd.ch.str.upper().isin(POST4)].groupby(["sub", "task"]).delta.mean().unstack().dropna()
        x = (q.restEO - q.restEC).values
        tw = p[(p.kind == "iaaft0") & p.ch.str.upper().isin(POST4)]
        tw = tw.merge(dd[["sub", "task", "ch"]], on=["sub", "task", "ch"]).groupby(["sub", "task"]).delta.mean().unstack()
        r = dict(n=len(x), eo_minus_ec_mean=float(x.mean()), dz=float(x.mean() / x.std(ddof=1)),
                 median_delta=dict(EC=float(q.restEC.median()), EO=float(q.restEO.median())),
                 median_twin=dict(EC=float(tw.restEC.median()), EO=float(tw.restEO.median())))
        mont = mne.channels.make_standard_montage("standard_1020").get_positions()["ch_pos"]
        pos = {k.upper(): v for k, v in mont.items()}; chans = list(EEG26)
        adj = channel_adjacency([c.upper() for c in chans], pos)
        w = dd.pivot_table(index=["sub", "ch"], columns="task", values="delta")
        D = (w["restEO"] - w["restEC"]).unstack("ch").reindex(columns=chans)
        t, cl = cluster_test(D.values, adj, n_perm=NPERM, seed=1)
        r["EO-EC delta clusters"] = [dict(sign=c["sign"], n_channels=len(c["channels"]), p=c["p"]) for c in cl if c["p"] < 0.05]
        res[tag] = r
    out["TDBRAIN EO-EC sensitivity"] = res
    # ANPHY-Sleep and Sleep-EDF spindle band
    sig = pd.read_csv("results/post_anphy_sigma.csv", usecols=AA.COLS + ["peak_height"])
    sig = flags(sig, "sigma"); sig["delta"] = fold_delta(sig.th_mu_q50, sig.th_c_q50); sig["covered"] = True
    pos = positions(); scalp = [c for c in sorted(sig.ch.unique()) if c in pos]; adj = channel_adjacency(scalp, pos)
    out["ANPHY spindle-band fractions (scalp)"] = frac(sig[(sig.kind == "data") & sig.ch.isin(scalp)], "stage").to_dict(orient="index")
    an = {}
    for tag, dd in (("all", sig), ("credible", sig[sig.credible | (sig.kind == "iaaft0")])):
        if tag == "credible":   # keep twins only for pieces whose data peak is credible
            ok = sig[(sig.kind == "data") & sig.credible][["rec", "ch", "stage", "chunk"]]
            dd = pd.concat([sig[(sig.kind == "data") & sig.credible], sig[sig.kind == "iaaft0"].merge(ok, on=["rec", "ch", "stage", "chunk"])])
        o, _ = AA.summarize(dd, scalp, adj)
        an[tag] = {"scalp mean delta by stage": {s: o["data|scalp mean"][s]["delta"] for s in AA.STAGES},
                   "twin scalp mean delta by stage": {s: o["twin|scalp mean"][s]["delta"] for s in AA.STAGES},
                   "N2-W delta clusters": [dict(sign=c["sign"], n_channels=len(c["channels"]), p=c["p"]) for c in o["N2-W|delta"]["clusters"] if c["p"] < 0.05],
                   "N2 data-twin delta clusters": [dict(sign=c["sign"], n_channels=len(c["channels"]), p=c["p"]) for c in o["N2 data-twin|delta"]["clusters"] if c["p"] < 0.05]}
    out["ANPHY N2-W sensitivity"] = an
    se = pd.read_csv("results/post_sleep_sigma.csv", usecols=["rec", "ch", "stage", "chunk", "kind", "peak", "peak_height", "th_mu_q50", "th_c_q50"])
    se = flags(se, "sigma"); se["delta"] = fold_delta(se.th_mu_q50, se.th_c_q50); se["subj"] = se.rec.str[:5]
    out["Sleep-EDF spindle-band fractions"] = {ch: frac(se[(se.kind == "data") & (se.ch == ch)], "stage").to_dict(orient="index") for ch in se.ch.unique()}
    sd = {}
    for tag, dd in (("all", se), ("credible", se[se.credible])):
        g = dd[(dd.kind == "data") & (dd.ch == "EEG Fpz-Cz")].groupby(["subj", "stage"]).delta.mean().unstack()
        x = (g.N2 - g.W).dropna()
        sd[tag] = dict(median_by_stage={s: float(g[s].median()) for s in AA.STAGES if s in g}, n_pairs=len(x),
                       n2_minus_w_dz=float(x.mean() / x.std(ddof=1)), frac_positive=float((x > 0).mean()))
    out["Sleep-EDF Fpz-Cz sensitivity"] = sd
    json.dump(out, open("results/peak_sensitivity.json", "w"), indent=1, default=str)
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()

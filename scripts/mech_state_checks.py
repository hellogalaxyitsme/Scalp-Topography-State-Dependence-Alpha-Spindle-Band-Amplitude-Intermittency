"""Checks of the empirical fast-modulation probabilities of mech_twin_broad.py against confounds of the state
contrasts: (1) the change in probability from eyes closed to eyes open (TDBRAIN alpha) and from wakefulness to N2
(spindle band) at matched estimated fold distance, by stratifying pieces into delta bins of width 0.5; (2) the same
contrasts and the near-fold probabilities restricted to pieces with a credible spectral peak; (3) for TDBRAIN,
restricted to channel recordings within the simulated support, and the eye-opening change at zero change in the
estimated signal-to-background ratio (intercept of the participant-level regression on the SNR change).
usage: python scripts/mech_state_checks.py  -> results/mech_state_checks.json"""
import json, numpy as np, pandas as pd
from scipy.stats import wilcoxon, t as tdist

PRIOR_F = {"alpha": (8.0, 12.5), "sigma": (11.5, 14.5)}
HMIN = 0.1
SETS = {"tdbrain": ("alpha", ["sub", "ses", "task", "ch"], "sub", "task", "restEC", "restEO"),
        "anphy_sigma": ("sigma", ["rec", "ch", "stage", "chunk"], "rec", "stage", "W", "N2"),
        "sleep_sigma": ("sigma", ["rec", "ch", "stage", "chunk"], "rec", "stage", "W", "N2")}
BINS = np.arange(-8.0, 3.01, 0.5)
NB = 1000
rng = np.random.default_rng(3)


def participant_change(d, unit, s, a, b):
    G = d.groupby([unit, s]).p_fast.mean().unstack()
    if a not in G or b not in G:
        return dict(n=0)
    g = G[[a, b]].dropna()
    if len(g) < 5:
        return dict(n=int(len(g)))
    return dict(n=int(len(g)), mean_a=float(g[a].mean()), mean_b=float(g[b].mean()), p=float(wilcoxon(g[b], g[a]).pvalue))


def matched_diff(d, s, a, b):
    """Difference b - a of the mean probability within delta bins, weighted by n_a n_b / (n_a + n_b)."""
    num = den = 0.0
    for _, g in d.groupby("bin", observed=True):
        xa, xb = g.p_fast[g[s] == a], g.p_fast[g[s] == b]
        if len(xa) and len(xb):
            w = len(xa) * len(xb) / (len(xa) + len(xb))
            num += w * (xb.mean() - xa.mean()); den += w
    return num / den if den else np.nan


def main():
    out = {}
    for t, (band, keys, unit, s, a, b) in SETS.items():
        M = pd.read_csv(f"results/mechtwb_{t}.csv")
        P = pd.read_csv(f"results/post_{t}.csv", usecols=keys + ["kind", "peak", "peak_height", "th_snr_db_hat"])
        if t == "tdbrain":
            P["covered"] = pd.read_csv("results/coverage_tdbrain.csv").cov_pct.values < 95
        P = P[P.kind == "data"].drop(columns="kind")
        lo, hi = PRIOR_F[band]
        P["credible"] = P.peak.between(lo, hi) & (P.peak_height >= HMIN)
        d = M.merge(P, on=keys, how="left")
        if t == "sleep_sigma":
            d["rec"] = d.rec.str[:5]
        d = d[d[s].isin([a, b])].copy()
        d["bin"] = pd.cut(d.delta, BINS)
        r = dict(all=participant_change(d, unit, s, a, b), credible=participant_change(d[d.credible == True], unit, s, a, b))
        units = d[unit].unique()
        est = matched_diff(d, s, a, b)
        bs = []
        groups = {u: g for u, g in d.groupby(unit)}
        for _ in range(NB):
            pick = rng.choice(units, len(units))
            bs.append(matched_diff(pd.concat([groups[u] for u in pick]), s, a, b))
        r["delta-matched difference"] = dict(estimate=float(est), ci=[float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))],
                                             raw_difference=float(d.p_fast[d[s] == b].mean() - d.p_fast[d[s] == a].mean()))
        near = d.delta > -1
        r["near fold"] = {lab: dict(n=int(m.sum()), mean=float(d.p_fast[m].mean()), frac_gt_0_5=float((d.p_fast[m] > 0.5).mean()))
                          for lab, m in (("all", near), ("credible", near & (d.credible == True)))}
        if t == "tdbrain":
            c = d[d.covered == True]
            r["within support"] = participant_change(c, unit, s, a, b)
            r["near fold"]["within support, credible"] = dict(n=int((near & (d.covered == True) & (d.credible == True)).sum()),
                                                               mean=float(d.p_fast[near & (d.covered == True) & (d.credible == True)].mean()))
            g = d.groupby([unit, s])[["p_fast", "th_snr_db_hat"]].mean().unstack().dropna()
            y = (g[("p_fast", b)] - g[("p_fast", a)]).values
            x = (g[("th_snr_db_hat", b)] - g[("th_snr_db_hat", a)]).values
            Z = np.column_stack([np.ones(len(x)), x])
            beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
            res = y - Z @ beta; s2 = res @ res / (len(y) - 2)
            se = np.sqrt(s2 * np.linalg.inv(Z.T @ Z)[0, 0])
            r["eye opening at zero SNR change"] = dict(n=int(len(y)), intercept=float(beta[0]), slope=float(beta[1]),
                                                       t=float(beta[0] / se), p=float(2 * tdist.sf(abs(beta[0] / se), len(y) - 2)),
                                                       raw_mean_change=float(y.mean()), median_snr_change=float(np.median(x)))
        out[t] = r
        print(t, json.dumps(r, indent=1), flush=True)
    json.dump(out, open("results/mech_state_checks.json", "w"), indent=1)


if __name__ == "__main__":
    main()

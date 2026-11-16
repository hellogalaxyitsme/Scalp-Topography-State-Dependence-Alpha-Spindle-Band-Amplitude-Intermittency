"""Scalp maps of the model coordinates and of direct statistics for the two state contrasts: eye opening (TDBRAIN
healthy adults, first session, 26 channels) and N2 minus wakefulness in the spindle band (ANPHY-Sleep, scalp
channels; pieces summarized by their median per night, channel and stage, as in analyze_anphy.py). For each measure:
the paired d_z map, its clusters (cluster permutation test as for delta), and the similarity of the map with the
delta map (Spearman correlation over channels; participant-bootstrap interval in which all maps are recomputed from
the same resampled participants). Direct statistics as in descriptor_benchmark.py, with the peak height above the
aperiodic fit (periodic power, log10 units) and the background fraction.
usage: python scripts/spatial_benchmark.py [n_perm]  -> results/spatial_benchmark.json"""
import sys, json, numpy as np, pandas as pd, mne
from scipy.stats import spearmanr
from ampint.io_tdbrain import load_meta, EEG26
from ampint.io_anphy import positions
from ampint.stats import channel_adjacency, cluster_test
from descriptor_benchmark import add_measures

NPERM = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
NB = 1000
rng = np.random.default_rng(8)
ACF = [f"a{i:02d}" for i in range(40)]
COMMON = ["delta", "th_mu_q50", "th_c_q50", "log_t0_q50", "cv", "kurt", "lq99", "tau_e", "peak_height", "rho"]


def dz_map(D):
    return np.nanmean(D, 0) / np.nanstd(D, 0, ddof=1)


def compare(A, chans, adj, seed):
    """A: dict measure -> participants x channels array of paired differences (same participant order)."""
    out = {}
    n = A["delta"].shape[0]
    reps = [rng.integers(0, n, n) for _ in range(NB)]
    ref = dz_map(A["delta"]); ref_b = [dz_map(A["delta"][i]) for i in reps]
    for m, D in A.items():
        e = dz_map(D)
        _, cl = cluster_test(D, adj, n_perm=NPERM, seed=seed)
        r = dict(n=int(np.isfinite(D).any(1).sum()), mean_dz=float(np.nanmean(e)), dz_map=dict(zip(chans, np.round(e, 3).tolist())),
                 clusters=[dict(sign=c["sign"], n_channels=len(c["channels"]), channels=[chans[i] for i in c["channels"]], p=c["p"])
                           for c in cl if c["p"] < 0.05])
        if m != "delta":
            rb = [spearmanr(rb_, dz_map(D[i]), nan_policy="omit")[0] for rb_, i in zip(ref_b, reps)]
            r["spearman with delta map"] = float(spearmanr(ref, e, nan_policy="omit")[0])
            r["spearman ci"] = [float(np.nanpercentile(rb, 2.5)), float(np.nanpercentile(rb, 97.5))]
        out[m] = r
    return out


def tdbrain():
    meta = load_meta(); m1 = meta[meta.sessID == 1].drop_duplicates("sub")
    keep = set(m1.loc[(m1.indication == "HEALTHY") & (m1.age >= 18) & m1.gender.notna(), "sub"])
    P = pd.read_csv("results/post_tdbrain.csv", usecols=["sub", "ses", "task", "ch", "kind", "th_mu_q50", "th_c_q50", "log_t0_q50",
                                                         "cv", "kurt", "lq99", "peak_height", "rho"] + ACF)
    P = add_measures(P[(P.kind == "data") & (P.ses == "ses-1") & P["sub"].isin(keep)].copy())
    chans = list(EEG26)
    mont = mne.channels.make_standard_montage("standard_1020").get_positions()["ch_pos"]
    pos = {k.upper(): v for k, v in mont.items()}
    adj = channel_adjacency([c.upper() for c in chans], pos)
    subs = sorted(P["sub"].unique())
    A = {}
    for m in COMMON:
        w = P.pivot_table(index=["sub", "ch"], columns="task", values=m)
        A[m] = (w["restEO"] - w["restEC"]).unstack("ch").reindex(index=subs, columns=chans).values
    return compare(A, chans, adj, seed=1)


def anphy():
    pos = positions()
    S = pd.read_csv("results/post_anphy_sigma.csv", usecols=["rec", "ch", "stage", "chunk", "kind", "th_mu_q50", "th_c_q50", "log_t0_q50",
                                                             "cv", "kurt", "lq99", "peak_height", "rho"] + ACF)
    S = add_measures(S[(S.kind == "data") & S.stage.isin(["W", "N2"])].copy())
    scalp = [c for c in sorted(S.ch.unique()) if c in pos]
    adj = channel_adjacency(scalp, pos)
    G = S[S.ch.isin(scalp)].groupby(["rec", "ch", "stage"])[COMMON].median()
    recs = sorted(S.rec.unique())
    A = {m: (G[m].unstack("stage").N2 - G[m].unstack("stage").W).unstack("ch").reindex(index=recs, columns=scalp).values for m in COMMON}
    return compare(A, scalp, adj, seed=3)


def main():
    out = {"eye opening, TDBRAIN": tdbrain(), "N2 minus wakefulness, ANPHY-Sleep": anphy()}
    json.dump(out, open("results/spatial_benchmark.json", "w"), indent=1)
    for k, v in out.items():
        print(k)
        for m, r in v.items():
            cl = "; ".join(f"{'+' if c['sign'] > 0 else '-'}{c['n_channels']}ch p={c['p']:.4f}" for c in r["clusters"])
            sim = f"rs={r['spearman with delta map']:+.2f} [{r['spearman ci'][0]:+.2f},{r['spearman ci'][1]:+.2f}]" if m != "delta" else ""
            print(f"  {m:14s} mean dz={r['mean_dz']:+.2f} {sim:28s} {cl}")


if __name__ == "__main__":
    main()

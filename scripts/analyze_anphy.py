"""ANPHY-Sleep replication of the sleep results with high-density EEG (average reference, plus the two Sleep-EDF
bipolar derivations): stage dependence of the spindle (sigma) coordinates, linear twins on matched chunks, and
scalp topography of N2 - wake with cluster-based permutation over nights. Also writes alpha medians per night,
channel and stage for the state-axis analysis.
usage: python scripts/analyze_anphy.py [n_perm]"""
import sys, os, json, numpy as np, pandas as pd
from ampint.nf import fold_delta
from ampint.io_anphy import positions
from ampint.stats import channel_adjacency, cluster_test

NPERM = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
VAR = sys.argv[2] if len(sys.argv) > 2 else ""          # "" or "csd" (surface Laplacian)
SFX = f"_{VAR}" if VAR else ""
STAGES = ["W", "N1", "N2", "N3", "REM"]
COLS = ["rec", "ch", "stage", "chunk", "kind", "peak", "rho", "th_mu_q50", "th_c_q50", "log_t0_q50", "p_bistable", "slow_flag"]
M = ["delta", "th_c_q50", "th_mu_q50", "log_t0_q50", "p_bistable", "peak"]


def load(band):
    d = pd.read_csv(f"results/post_anphy{SFX}_{band}.csv", usecols=COLS)
    cf = f"results/coverage_anphy{SFX}_{band}.csv"
    d["covered"] = pd.read_csv(cf).cov_pct.values < 95 if os.path.exists(cf) else True
    d["delta"] = fold_delta(d.th_mu_q50, d.th_c_q50)
    return d


def night_stage(d):
    return d.groupby(["rec", "ch", "stage"])[M].median().reset_index()


def summarize(d, scalp, adj):
    out = {}
    tw = d[d.kind == "iaaft0"]
    dm = d[d.kind == "data"].merge(tw[["rec", "ch", "stage", "chunk"]], on=["rec", "ch", "stage", "chunk"])  # chunks with twins
    for name, sub in (("data", d[d.kind == "data"]), ("data_twin_chunks", dm), ("twin", tw)):
        S = night_stage(sub)
        for region, chs in (("scalp mean", scalp), ("FPZ-CZ", ["FPZ-CZ"]), ("PZ-OZ", ["PZ-OZ"])):
            r = S[S.ch.isin(chs)].groupby(["rec", "stage"])[M].mean().reset_index()
            out[f"{name}|{region}"] = {st: dict(n=int((r.stage == st).sum()), **{k: float(r[r.stage == st][k].median()) for k in M})
                                       for st in STAGES}
    # N2 - W and N2 data - twin per scalp channel, paired over nights
    S = night_stage(d[d.kind == "data"])
    T = night_stage(tw); Dm = night_stage(dm)
    for k in ["delta", "th_c_q50", "p_bistable", "log_t0_q50"]:
        w = S.pivot_table(index=["rec", "ch"], columns="stage", values=k)
        D = (w["N2"] - w["W"]).unstack("ch").reindex(columns=scalp)
        t, cl = cluster_test(D.values, adj, n_perm=NPERM, seed=3)
        out[f"N2-W|{k}"] = dict(n_nights=int(D.notna().any(axis=1).sum()), mean=dict(zip(scalp, np.round(np.nanmean(D.values, 0), 4).tolist())),
                               dz=dict(zip(scalp, np.round(np.nanmean(D.values, 0) / np.nanstd(D.values, 0, ddof=1), 3).tolist())),
                               clusters=[dict(c, channels=[scalp[i] for i in c["channels"]]) for c in cl])
    for k in ["delta", "p_bistable"]:
        a = Dm[Dm.stage == "N2"].pivot_table(index="rec", columns="ch", values=k).reindex(columns=scalp)
        b = T[T.stage == "N2"].pivot_table(index="rec", columns="ch", values=k).reindex(index=a.index, columns=scalp)
        t, cl = cluster_test((a - b).values, adj, n_perm=NPERM, seed=4)
        out[f"N2 data-twin|{k}"] = dict(n_nights=int(a.notna().any(axis=1).sum()), mean=dict(zip(scalp, np.round(np.nanmean((a - b).values, 0), 4).tolist())),
                                       clusters=[dict(c, channels=[scalp[i] for i in c["channels"]]) for c in cl])
    med = S.groupby(["ch", "stage"])[M].median().reset_index()
    return out, med


def main():
    pos = positions()
    sig = load("sigma")
    scalp = [c for c in sorted(sig.ch.unique()) if c in pos]
    adj = channel_adjacency(scalp, pos)
    res = dict(n_nights=int(sig.rec.nunique()), n_scalp_channels=len(scalp), n_neighbours_median=float(np.median(adj.sum(1))))
    for tag, d in (("all", sig), ("covered", sig[sig.covered])):
        out, med = summarize(d, scalp, adj)
        res[tag] = out
        if tag == "all":
            med.to_csv(f"results/anphy{SFX}_sigma_channel_stage.csv", index=False)
        for k, v in out.items():
            if "clusters" in v:
                cl = [f"{'+' if c['sign'] > 0 else '-'}{len(c['channels'])}ch p={c['p']:.4f}" for c in v["clusters"] if c["p"] < 0.05]
                print(f"[{tag}] {k:24s} n={v['n_nights']} {cl}")
            else:
                print(f"[{tag}] {k:28s} " + "  ".join(f"{st}: d={v[st]['delta']:+.2f} c={v[st]['th_c_q50']:.2f} Pb={v[st]['p_bistable']:.2f} (n={v[st]['n']})" for st in STAGES))
    json.dump(res, open(f"results/anphy{SFX}_summary.json", "w"), indent=1)
    al = load("alpha")
    al = al[al.kind == "data"].groupby(["rec", "ch", "stage"])[["th_mu_q50", "th_c_q50", "log_t0_q50", "p_bistable", "delta"]].median().reset_index()
    al.to_csv(f"results/anphy{SFX}_alpha_subject_stage.csv", index=False)
    print("alpha rows", len(al))


if __name__ == "__main__":
    main()

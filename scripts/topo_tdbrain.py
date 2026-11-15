"""Scalp topography of the normal-form coordinates of the alpha rhythm (TDBRAIN V3.1, first session, adults).
Healthy eyes-closed maps; within-participant eyes-open minus eyes-closed maps, also adjusted per channel for
the change in estimated SNR; MDD minus healthy maps (age/sex-normalised; recorded reference only).
Channel-level inference by cluster-based permutation over the 26-channel layout.
Sensitivity: rows outside the simulated support removed (recorded reference); variants 'eog' (EOG regressed
out) and 'csd' (surface Laplacian), computed for healthy adults (run_nf_tdbrain_variants.py).
usage: python scripts/topo_tdbrain.py [n_perm] [variant]"""
import sys, json, numpy as np, pandas as pd, mne
from ampint.io_tdbrain import load_meta, EEG26
from ampint.nf import fold_delta
from ampint.stats import channel_adjacency, cluster_test, design

NPERM = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
VAR = sys.argv[2] if len(sys.argv) > 2 else ""
SFX = f"_{VAR}" if VAR else ""
MEAS = ["delta", "th_mu_q50", "th_c_q50", "log_t0_q50", "p_bistable"]
PRIOR90 = dict(th_mu=0.9 * 11.0, th_c=0.9 * 12.0)   # width of the central 90% of the uniform priors


def clusters_named(cl, chans):
    return [dict(c, channels=[chans[i] for i in c["channels"]]) for c in cl]


def main():
    post = pd.read_csv(f"results/post_tdbrain{SFX}.csv", usecols=lambda c: not c.startswith(("h", "a0", "a1", "a2", "a3", "lq", "acf")))
    if VAR:
        post["covered"] = True
    else:
        cov = pd.read_csv("results/coverage_tdbrain.csv").cov_pct.values
        assert len(cov) == len(post)
        post["covered"] = cov < 95
    d = post[(post.kind == "data") & (post.ses == "ses-1")].copy()
    d["delta"] = fold_delta(d.th_mu_q50, d.th_c_q50)
    d["info_mu"] = 1 - (d.th_mu_hi90 - d.th_mu_lo90) / PRIOR90["th_mu"]
    d["info_c"] = 1 - (d.th_c_hi90 - d.th_c_lo90) / PRIOR90["th_c"]
    meta = load_meta()
    m1 = meta[meta.sessID == 1].drop_duplicates("sub")[["sub", "indication", "age", "gender"]]
    d = d.merge(m1, on="sub", how="left")
    d = d[(d.age >= 18) & d.gender.notna()]
    mont = mne.channels.make_standard_montage("standard_1020").get_positions()["ch_pos"]
    pos = {k.upper(): v for k, v in mont.items()}
    chans = list(EEG26)
    adj = channel_adjacency([c.upper() for c in chans], pos)
    out = dict(variant=VAR or "recorded reference", channels=chans, n_neighbours=adj.sum(1).tolist(), n_perm=NPERM)
    rows = []
    for covered in ((False,) if VAR else (False, True)):
        dd = d[d.covered] if covered else d
        tag = "covered" if covered else "all"
        H = dd[dd.indication == "HEALTHY"]
        res = {}
        ec = H[H.task == "restEC"].groupby("ch")[MEAS + ["info_mu", "info_c", "rho", "peak", "th_snr_db_hat"]].median()
        for ch in chans:
            if ch in ec.index:
                rows.append(dict(sample=tag, contrast="healthy EC median", ch=ch, **ec.loc[ch].to_dict()))
        for k in MEAS:
            w = H.pivot_table(index=["sub", "ch"], columns="task", values=k).dropna()
            D = (w["restEO"] - w["restEC"]).unstack("ch").reindex(columns=chans)
            t, cl = cluster_test(D.values, adj, n_perm=NPERM, seed=1)
            res[f"EO-EC|{k}"] = dict(n=int(D.notna().any(axis=1).sum()), t=dict(zip(chans, np.round(t, 3).tolist())),
                                    mean=dict(zip(chans, np.round(np.nanmean(D.values, 0), 4).tolist())),
                                    dz=dict(zip(chans, np.round(np.nanmean(D.values, 0) / np.nanstd(D.values, 0, ddof=1), 3).tolist())),
                                    clusters=clusters_named(cl, chans))
        # fold distance change at zero change in estimated SNR (per-channel regression on the SNR change)
        w = H.pivot_table(index=["sub", "ch"], columns="task", values=["delta", "th_snr_db_hat"]).dropna()
        dlt = (w[("delta", "restEO")] - w[("delta", "restEC")]).unstack("ch").reindex(columns=chans)
        dsn = (w[("th_snr_db_hat", "restEO")] - w[("th_snr_db_hat", "restEC")]).unstack("ch").reindex(columns=chans)
        Dadj = dlt.copy()
        for ch in chans:
            ok = dlt[ch].notna() & dsn[ch].notna()
            b = np.polyfit(dsn[ch][ok], dlt[ch][ok], 1)[0]
            Dadj[ch] = dlt[ch] - b * dsn[ch]
        t, cl = cluster_test(Dadj.values, adj, n_perm=NPERM, seed=5)
        res["EO-EC|delta, SNR-adjusted"] = dict(n=int(Dadj.notna().any(axis=1).sum()), t=dict(zip(chans, np.round(t, 3).tolist())),
                                               mean=dict(zip(chans, np.round(np.nanmean(Dadj.values, 0), 4).tolist())),
                                               dz=dict(zip(chans, np.round(np.nanmean(Dadj.values, 0) / np.nanstd(Dadj.values, 0, ddof=1), 3).tolist())),
                                               clusters=clusters_named(cl, chans))
        if not VAR:  # MDD - healthy (eyes closed), residuals of a healthy age/sex model per channel
            E = dd[(dd.task == "restEC") & dd.indication.isin(["HEALTHY", "MDD"])].copy()
            for k in ["delta", "th_mu_q50", "th_c_q50", "log_t0_q50"]:
                E[k + "_res"] = np.nan
                for ch in chans:
                    m = E.ch == ch; ref = m & (E.indication == "HEALTHY")
                    if ref.sum() < 30:
                        continue
                    b = np.linalg.lstsq(design(E[ref]), E.loc[ref, k].values, rcond=None)[0]
                    E.loc[m, k + "_res"] = E.loc[m, k].values - design(E[m]) @ b
                W = E.pivot_table(index=["indication", "sub"], columns="ch", values=k + "_res").reindex(columns=chans)
                A, B = W.loc["MDD"].values, W.loc["HEALTHY"].values
                t, cl = cluster_test(A, adj, B=B, n_perm=NPERM, seed=2)
                res[f"MDD-healthy|{k}"] = dict(n=f"{len(A)}/{len(B)}", t=dict(zip(chans, np.round(t, 3).tolist())),
                                              diff=dict(zip(chans, np.round(np.nanmean(A, 0) - np.nanmean(B, 0), 4).tolist())),
                                              clusters=clusters_named(cl, chans))
        out[tag] = res
        for key, v in res.items():
            sig = [c for c in v["clusters"] if c["p"] < 0.05]
            print(f"[{VAR or 'ref'}|{tag}] {key:28s} n={v['n']}  " + "; ".join(f"{'+' if c['sign'] > 0 else '-'}{len(c['channels'])}ch p={c['p']:.4f}" for c in sig))
    pd.DataFrame(rows).to_csv(f"results/topo_tdbrain{SFX}_channels.csv", index=False)
    json.dump(out, open(f"results/topo_tdbrain{SFX}.json", "w"), indent=1)


if __name__ == "__main__":
    main()

"""Eyes-open minus eyes-closed change in the fold distance at zero change in the estimated signal-to-background
ratio (TDBRAIN healthy adults, first session). Per channel, the paired change in delta is regressed on the paired
change in the estimated signal-to-background ratio (posterior median of the simulated SNR, dB); the intercept is
tested with a cluster-based Freedman-Lane sign-flipping scheme (ampint.stats.cluster_test_cov), so that the
uncertainty of the slope enters the null distribution. Recorded reference, ocular regression and surface Laplacian.
usage: python scripts/snr_adjusted_topo.py [n_perm]  -> results/snradj_tdbrain.json"""
import sys, json, numpy as np, pandas as pd, mne
from ampint.io_tdbrain import load_meta, EEG26
from ampint.nf import fold_delta
from ampint.stats import channel_adjacency, cluster_test_cov

NPERM = int(sys.argv[1]) if len(sys.argv) > 1 else 5000


def main():
    meta = load_meta()
    m1 = meta[meta.sessID == 1].drop_duplicates("sub")[["sub", "indication", "age", "gender"]]
    mont = mne.channels.make_standard_montage("standard_1020").get_positions()["ch_pos"]
    pos = {k.upper(): v for k, v in mont.items()}
    chans = list(EEG26)
    adj = channel_adjacency([c.upper() for c in chans], pos)
    out = dict(n_perm=NPERM, channels=chans, covariate="paired change in the estimated signal-to-background ratio (dB)")
    for var, lab in (("", "recorded reference"), ("eog", "ocular regression"), ("csd", "surface Laplacian")):
        post = pd.read_csv(f"results/post_tdbrain{'_' + var if var else ''}.csv",
                           usecols=["sub", "ses", "task", "ch", "kind", "th_mu_q50", "th_c_q50", "th_snr_db_hat"])
        d = post[(post.kind == "data") & (post.ses == "ses-1")].merge(m1, on="sub", how="left")
        H = d[(d.age >= 18) & d.gender.notna() & (d.indication == "HEALTHY")].copy()
        H["delta"] = fold_delta(H.th_mu_q50.values, H.th_c_q50.values)
        w = H.pivot_table(index=["sub", "ch"], columns="task", values=["delta", "th_snr_db_hat"]).dropna()
        dlt = (w[("delta", "restEO")] - w[("delta", "restEC")]).unstack("ch").reindex(columns=chans)
        dsn = (w[("th_snr_db_hat", "restEO")] - w[("th_snr_db_hat", "restEC")]).unstack("ch").reindex(columns=chans)
        r, cl = cluster_test_cov(dlt.values, dsn.values, adj, n_perm=NPERM, seed=5)
        r["n_per_channel"] = r.pop("n")
        out[lab] = dict(n=int(dlt.notna().any(axis=1).sum()),
                        snr_change_db_median=float(np.nanmedian(dsn.values)),
                        **{k: dict(zip(chans, np.round(v, 4).tolist())) for k, v in r.items()},
                        clusters=[dict(c, channels=[chans[i] for i in c["channels"]]) for c in cl])
        sig = [c for c in cl if c["p"] < 0.05]
        print(f"{lab:20s} n={out[lab]['n']}  " + "; ".join(f"{'+' if c['sign'] > 0 else '-'}{len(c['channels'])}ch p={c['p']:.4f}" for c in sig))
    json.dump(out, open("results/snradj_tdbrain.json", "w"), indent=1)


if __name__ == "__main__":
    main()

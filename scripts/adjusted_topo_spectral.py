"""Eye opening at zero change of directly measured spectral quantities (TDBRAIN healthy adults, first session,
recorded reference). As snr_adjusted_topo.py, with the paired change in the peak height above the aperiodic fit
(log10 units) or in the background fraction in place of the estimated SNR. Also the held-out recovery of the
estimated SNR (posterior median of the simulated signal-to-background ratio) in the alpha and spindle-band banks.
usage: python scripts/adjusted_topo_spectral.py [n_perm]  -> results/adjusted_topo_spectral.json"""
import sys, json, numpy as np, pandas as pd, mne
from scipy.stats import spearmanr
from ampint.io_tdbrain import load_meta, EEG26
from ampint.nf import fold_delta
from ampint.stats import channel_adjacency, cluster_test_cov

NPERM = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
BOX = ((-7.0, -3.0), (1.5, 5.5))


def main():
    meta = load_meta()
    m1 = meta[meta.sessID == 1].drop_duplicates("sub")[["sub", "indication", "age", "gender"]]
    mont = mne.channels.make_standard_montage("standard_1020").get_positions()["ch_pos"]
    pos = {k.upper(): v for k, v in mont.items()}
    chans = list(EEG26)
    adj = channel_adjacency([c.upper() for c in chans], pos)
    post = pd.read_csv("results/post_tdbrain.csv", usecols=["sub", "ses", "task", "ch", "kind", "th_mu_q50", "th_c_q50", "peak_height", "rho"])
    d = post[(post.kind == "data") & (post.ses == "ses-1")].merge(m1, on="sub", how="left")
    H = d[(d.age >= 18) & d.gender.notna() & (d.indication == "HEALTHY")].copy()
    H["delta"] = fold_delta(H.th_mu_q50.values, H.th_c_q50.values)
    w = H.pivot_table(index=["sub", "ch"], columns="task", values=["delta", "peak_height", "rho"]).dropna()
    dlt = (w[("delta", "restEO")] - w[("delta", "restEC")]).unstack("ch").reindex(columns=chans)
    out = dict(n_perm=NPERM, channels=chans, n=int(dlt.notna().any(axis=1).sum()))
    for cov, lab in (("peak_height", "change in peak height"), ("rho", "change in background fraction")):
        x = (w[(cov, "restEO")] - w[(cov, "restEC")]).unstack("ch").reindex(columns=chans)
        r, cl = cluster_test_cov(dlt.values, x.values, adj, n_perm=NPERM, seed=5)
        out[lab] = dict(covariate_change_median=float(np.nanmedian(x.values)), frac_positive=float(np.nanmean(x.values > 0)),
                        es=dict(zip(chans, np.round(r["es"], 3).tolist())),
                        clusters=[dict(sign=c["sign"], n_channels=len(c["channels"]), channels=[chans[i] for i in c["channels"]], p=c["p"])
                                  for c in cl if c["p"] < 0.05])
        print(lab, out[lab]["covariate_change_median"], [(c["sign"], c["n_channels"], c["p"]) for c in out[lab]["clusters"]], flush=True)
    rec = {}
    for cfg in ("alpha", "sigma250"):
        Hh = pd.read_csv(f"results/sbi_{cfg}_full_heldout.csv", usecols=["th_snr_db", "th_snr_db_hat", "th_mu", "th_c"])
        box = (Hh.th_mu.between(*BOX[0]) & Hh.th_c.between(*BOX[1])).values
        for lab, m in (("whole prior", np.ones(len(Hh), bool)), ("healthy region", box)):
            y, p = Hh.th_snr_db.values[m], Hh.th_snr_db_hat.values[m]
            rec[f"{cfg}, {lab}"] = dict(n=int(m.sum()), r2=float(1 - np.mean((p - y) ** 2) / np.var(y)), rs=float(spearmanr(y, p)[0]),
                                        mae_db=float(np.mean(np.abs(p - y))))
    out["held-out recovery of the estimated SNR"] = rec
    print(json.dumps(rec, indent=1))
    json.dump(out, open("results/adjusted_topo_spectral.json", "w"), indent=1)


if __name__ == "__main__":
    main()

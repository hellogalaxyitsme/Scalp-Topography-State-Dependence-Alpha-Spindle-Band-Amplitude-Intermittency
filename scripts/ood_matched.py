"""Coverage of empirical summaries by the simulation bank, matched in estimated parameters.
For every empirical summary vector, the held-out simulation with the closest estimated parameters
(posterior medians of mu, c, log t0, nuisance SNR and exponent, measured peak and record length) serves as a
reference; the k-NN distance of the empirical vector to the training bank is compared with the distribution
of the same distance for these matched simulations."""
import json, numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
from sbi_fit import load_bank
from ampint.summ import FEATURES

PAR = ["th_mu_q50", "th_c_q50", "log_t0_q50", "th_snr_db_hat", "th_chi_hat", "peak", "T"]


def run(cfg, emp_sets):
    df = load_bank(cfg)
    u = np.random.default_rng(0).random(len(df))
    tr = df[u < 0.8]
    H = pd.read_csv(f"results/sbi_{cfg}_full_heldout.csv")
    mu, sd = tr[FEATURES].mean(), tr[FEATURES].std().replace(0, 1)
    Z = lambda d: ((d[FEATURES] - mu) / sd).values
    nn = NearestNeighbors(n_neighbors=10).fit(Z(tr.sample(min(40000, len(tr)), random_state=0)))
    H["knn"] = nn.kneighbors(Z(H))[0].mean(1)
    pm, ps = H[PAR].mean(), H[PAR].std()
    pnn = NearestNeighbors(n_neighbors=20).fit(((H[PAR] - pm) / ps).values)
    out = {}
    for name, e in emp_sets.items():
        e = e.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + PAR)
        e = e.sample(min(20000, len(e)), random_state=0)
        d_e = nn.kneighbors(Z(e))[0].mean(1)
        idx = pnn.kneighbors(((e[PAR] - pm) / ps).values)[1]
        d_ref = H["knn"].values[idx]                      # 20 parameter-matched simulations per empirical vector
        pct = (d_ref < d_e[:, None]).mean(1) * 100        # percentile of the empirical distance among its matches
        ratio = d_e / np.median(d_ref, 1)
        out[name] = dict(n=int(len(e)), median_percentile=float(np.median(pct)),
                         frac_above_p95=float(np.mean(pct >= 95)), frac_above_all=float(np.mean(pct >= 100)),
                         median_distance_ratio=float(np.median(ratio)), frac_ratio_gt2=float(np.mean(ratio > 2)))
        print(cfg, name, out[name], flush=True)
    return out


if __name__ == "__main__":
    cols = FEATURES + PAR
    POST = ["O1", "Oz", "O2", "Pz", "P3", "P4"]
    tb = pd.read_csv("results/post_tdbrain.csv", usecols=cols + ["kind", "ch"])
    mz = pd.read_csv("results/post_mumtaz.csv", usecols=cols + ["ch"])
    dd = pd.read_csv("results/post_ds003478.csv", usecols=cols + ["ch"])
    sa = pd.read_csv("results/post_sleep_alpha.csv", usecols=cols + ["kind"])
    ss = pd.read_csv("results/post_sleep_sigma.csv", usecols=cols + ["kind"])
    res = {"alpha": run("alpha", {"TDBRAIN": tb[(tb.kind == "data") & tb.ch.isin(POST)],
                                  "Mumtaz": mz[mz.ch.isin(["O1", "O2", "P3", "P4", "Pz"])],
                                  "ds003478": dd[dd.ch.isin(["O1", "OZ", "O2", "PZ", "P3", "P4", "POZ", "PO3", "PO4"])],
                                  "Sleep-EDF alpha": sa[sa.kind == "data"]}),
           "sigma250": run("sigma250", {"Sleep-EDF sigma": ss[ss.kind == "data"]})}
    json.dump(res, open("results/ood_matched.json", "w"), indent=1)

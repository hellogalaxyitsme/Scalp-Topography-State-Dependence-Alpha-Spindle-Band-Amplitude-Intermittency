"""Burst predictions in simulations with known parameters: held-out alpha-setting simulations from the healthy-alpha
region are simulated again at their true parameters and at their posterior medians (with the estimated background
exponent, SNR and peak frequency, as in ppc_bursts.py), cut into 20-s runs, and their burst statistics compared.
usage: python scripts/ppc_recovery.py <n_jobs> <n_sims> [n_seeds]  -> results/bursts_ppc_recovery.csv"""
import sys, numpy as np, pandas as pd
from joblib import Parallel, delayed
from sbi_bank import observe, CONFIGS
from ampint.bursts import all_stats, split_like

NS = int(sys.argv[3]) if len(sys.argv) > 3 else 2
BOX = ((-7.0, -3.0), (1.5, 5.5))
RUN = 20.0


def one(i, r):
    cfg = CONFIGS["alpha"]
    L = np.full(int(r.th_T // RUN), int(RUN * cfg["fs"]))
    pars = {"true": dict(mu=r.th_mu, c=r.th_c, t0=r.th_t0, chi=r.th_chi, snr_db=r.th_snr_db, f0=r.th_f0),
            "median": dict(mu=r.th_mu_q50, c=r.th_c_q50, t0=10 ** r.log_t0_q50, chi=r.th_chi_hat, snr_db=r.th_snr_db_hat, f0=r.peak)}
    out = []
    for kind, th in pars.items():
        for k in range(NS):
            try:
                s, P = observe(cfg, dict(th, T=L.sum() / cfg["fs"] + 0.5), seed=50_000_000 + 1000 * i + 10 * k + (kind == "median"),
                               return_power=True)
                out.append(dict(sim=i, kind=kind, seed=k, **all_stats(split_like(P, L), cfg["fs"], s["peak"])))
            except Exception:
                pass
    return out


if __name__ == "__main__":
    H = pd.read_csv("results/sbi_alpha_full_heldout.csv")
    H = H[H.th_mu.between(*BOX[0]) & H.th_c.between(*BOX[1])]
    H = H.sample(min(int(sys.argv[2]), len(H)), random_state=0).reset_index(drop=True)
    print(len(H), "simulations", flush=True)
    res = Parallel(n_jobs=int(sys.argv[1]), verbose=5)(delayed(one)(i, r) for i, r in H.iterrows())
    pd.DataFrame([x for rr in res for x in rr]).to_csv("results/bursts_ppc_recovery.csv", index=False)

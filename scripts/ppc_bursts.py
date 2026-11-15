"""Posterior predictive burst statistics: for each channel-recording, the stochastic Bautin model is simulated at
the posterior median coordinates (with the estimated 1/f background, SNR and peak frequency, observation
pipeline as in the simulation bank), cut into the recording's clean-run lengths, and its burst statistics
computed exactly as for the EEG. Burst statistics were not used for inference.
usage: python scripts/ppc_bursts.py <n_jobs> [n_seeds]           healthy, first session (bursts_tdbrain)
       python ppc_bursts.py <n_jobs> [n_seeds] retest    session-1 posterior, session-2 run lengths (bursts_retest)
       a fourth argument adds a suffix to the output file (e.g. 'ppc_bursts.py 8 16 main ns16'); seeds 0..3 are
       the same in every run, so a larger n_seeds extends the original realizations."""
import sys, glob, numpy as np, pandas as pd
from joblib import Parallel, delayed
from sbi_bank import observe, CONFIGS
from ampint.bursts import all_stats, split_like

NS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
RETEST = len(sys.argv) > 3 and sys.argv[3] == "retest"
TAG = f"_{sys.argv[4]}" if len(sys.argv) > 4 else ""
PAR = ["th_mu_q50", "th_c_q50", "log_t0_q50", "th_chi_hat", "th_snr_db_hat"]


def one(i, r):
    cfg = CONFIGS["alpha"]
    L = np.array([int(float(v)) for v in str(r.run_lengths).split("|")])
    th = dict(mu=r.th_mu_q50, c=r.th_c_q50, t0=10 ** r.log_t0_q50, chi=r.th_chi_hat, snr_db=r.th_snr_db_hat,
              f0=r.peak, T=L.sum() / cfg["fs"] + 0.5)
    out = []
    for k in range(NS):
        try:
            s, P = observe(cfg, th, seed=(40_000_000 if RETEST else 20_000_000) + 100 * i + k, return_power=True)
            out.append(dict(sub=r["sub"], task=r.task, ch=r.ch, seed=k, sim_peak=s["peak"],
                            **all_stats(split_like(P, L), cfg["fs"], s["peak"])))
        except Exception:
            pass
    return out


if __name__ == "__main__":
    src = "results/bursts_retest" if RETEST else "results/bursts_tdbrain"
    b = pd.concat([pd.read_csv(f) for f in glob.glob(f"{src}/*.csv") if open(f).readline().strip()])
    b = b[(b.kind == "data") & ((b.ses == "ses-2") if RETEST else True)].drop(columns=["ses"], errors="ignore")
    post = pd.read_csv("results/post_tdbrain.csv", usecols=["sub", "ses", "task", "ch", "kind", "peak"] + PAR)
    post = post[(post.kind == "data") & (post.ses == "ses-1")].drop(columns=["kind", "ses"])
    if RETEST:  # the prediction uses session-1 coordinates and peak frequency; only the run lengths come from session 2
        b = b.drop(columns=["peak"])
    else:
        post = post.drop(columns=["peak"])
    d = b.merge(post, on=["sub", "task", "ch"], how="inner").dropna(subset=PAR).reset_index(drop=True)
    print(len(d), "channel-recordings", flush=True)
    res = Parallel(n_jobs=int(sys.argv[1]), verbose=5)(delayed(one)(i, r) for i, r in d.iterrows())
    pd.DataFrame([x for rr in res for x in rr]).to_csv(f"results/bursts_ppc{'_retest' if RETEST else ''}{TAG}.csv", index=False)

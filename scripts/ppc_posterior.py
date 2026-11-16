"""Posterior predictive burst statistics with parameter uncertainty: as ppc_bursts.py, but each realization uses
coordinates drawn from the recording's posterior. For mu_t, c_t and log10 t0, a value is drawn from the piecewise-
linear quantile function through the posterior quantiles (5, 25, 50, 75 and 95%, extrapolated linearly to 0 and 100%),
with independent draws for the three coordinates; background exponent, SNR and peak frequency are kept at their
estimates.
usage: python scripts/ppc_posterior.py <n_jobs> <n_draws> [retest]
       -> results/bursts_ppc_post.csv or results/bursts_ppc_post_retest.csv"""
import sys, glob, numpy as np, pandas as pd
from joblib import Parallel, delayed
from sbi_bank import observe, CONFIGS
from ampint.bursts import all_stats, split_like

ND = int(sys.argv[2])
RETEST = len(sys.argv) > 3 and sys.argv[3] == "retest"
Q = np.array([0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
TGT = {"mu": "th_mu", "c": "th_c", "lt0": "log_t0"}


def qcols(t):
    return [f"{t}_q{q}" for q in (5, 25, 50, 75, 95)]


def draw(r, t, u):
    q = np.array([r[c] for c in qcols(t)], float)
    q = np.r_[q[0] - (q[1] - q[0]) * 0.25, q, q[4] + (q[4] - q[3]) * 0.25]
    return float(np.interp(u, Q, q))


def one(i, r):
    cfg = CONFIGS["alpha"]
    L = np.array([int(float(v)) for v in str(r.run_lengths).split("|")])
    rng = np.random.default_rng((80_000_000 if RETEST else 60_000_000) + i)
    out = []
    for k in range(ND):
        u = rng.random(3)
        th = dict(mu=draw(r, "th_mu", u[0]), c=draw(r, "th_c", u[1]), t0=10 ** draw(r, "log_t0", u[2]),
                  chi=r.th_chi_hat, snr_db=r.th_snr_db_hat, f0=r.peak, T=L.sum() / cfg["fs"] + 0.5)
        try:
            s, P = observe(cfg, th, seed=(90_000_000 if RETEST else 70_000_000) + 1000 * i + k, return_power=True)
            out.append(dict(sub=r["sub"], task=r.task, ch=r.ch, draw=k, sim_peak=s["peak"],
                            **all_stats(split_like(P, L), cfg["fs"], s["peak"])))
        except Exception:
            pass
    return out


if __name__ == "__main__":
    src = "results/bursts_retest" if RETEST else "results/bursts_tdbrain"
    b = pd.concat([pd.read_csv(f) for f in glob.glob(f"{src}/*.csv") if open(f).readline().strip()])
    b = b[(b.kind == "data") & ((b.ses == "ses-2") if RETEST else True)].drop(columns=["ses"], errors="ignore")
    par = sum((qcols(t) for t in TGT.values()), []) + ["th_chi_hat", "th_snr_db_hat"]
    post = pd.read_csv("results/post_tdbrain.csv", usecols=["sub", "ses", "task", "ch", "kind", "peak"] + par)
    post = post[(post.kind == "data") & (post.ses == "ses-1")].drop(columns=["kind", "ses"])
    if RETEST:
        b = b.drop(columns=["peak"])
    else:
        post = post.drop(columns=["peak"])
    d = b.merge(post, on=["sub", "task", "ch"], how="inner").dropna(subset=par).reset_index(drop=True)
    print(len(d), "channel-recordings x", ND, "draws", flush=True)
    res = Parallel(n_jobs=int(sys.argv[1]), verbose=5)(delayed(one)(i, r) for i, r in d.iterrows())
    pd.DataFrame([x for rr in res for x in rr]).to_csv(f"results/bursts_ppc_post{'_retest' if RETEST else ''}.csv", index=False)

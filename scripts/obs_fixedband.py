"""Calibration of the fixed 11.5-14.5 Hz spindle-band analysis (sleep_checks.py), whose summaries are passed to a
posterior trained with peak-centred bands. Fresh simulations from the spindle-band bank priors (single source,
contiguous 60-115 s records) are summarized twice, with the peak-centred band of the bank (peak +- 1.5 Hz) and with
the fixed band, both with 1 s trimmed at each edge, and passed through the spindle-band posterior.
Reported: recovery of the true fold coordinate for both versions, the paired difference fixed - peak-centred, and,
as in the fold calibration, how often simulations with true delta < -3 or near the fold reach the observed
fixed-band N2 estimates.
usage: python scripts/obs_fixedband.py sims <n> <n_jobs>  -> results/obs_fixedband.csv
       python scripts/obs_fixedband.py summary            -> results/obs_fixedband.json"""
import sys, json, joblib, numpy as np, pandas as pd
from joblib import Parallel, delayed
from sbi_bank import CONFIGS, PRIOR
from sbi_twinbank import signal
from sbi_fit import predict
from ampint.nf import fold_delta
from ampint.spec import analytic, noise_fraction, peak_freq, psd
from ampint.summ import summaries

CFG = CONFIGS["sigma250"]
FIXED = (11.5, 14.5)
OFFSET = 95_000_000
N2_FIXED = {"ANPHY-Sleep scalp mean": -0.12, "Sleep-EDF Fpz-Cz": -0.02}   # median N2 estimates, fixed band


def one(k):
    seed = OFFSET + k
    rng = np.random.default_rng(seed)
    th = dict(mu=rng.uniform(*PRIOR["mu"]), c=rng.uniform(*PRIOR["c"]), t0=10 ** rng.uniform(*PRIOR["log10_t0"]),
              chi=rng.uniform(*PRIOR["chi"]), snr_db=rng.uniform(*PRIOR["snr_db"]), f0=rng.uniform(*CFG["f0"]),
              T=rng.uniform(*CFG["T"]))
    try:
        x = signal(CFG, th, seed, None, 1, 0.0)
        fs = CFG["fs"]; tr = int(fs)
        f, S = psd(x, fs)
        pk, _ = peak_freq(f, S, CFG["search"])
        rows = []
        for v, (lo, hi) in (("peak", (pk - CFG["hw"], pk + CFG["hw"])), ("fixed", FIXED)):
            rho = noise_fraction(f, S, lo, hi, fs)
            s = summaries([np.abs(analytic(x, lo, hi, fs)[tr:-tr]) ** 2], rho, fs)
            rows.append(dict(k=k, seed=seed, version=v, peak=pk, **{"th_" + a: b for a, b in th.items()}, **s))
        return rows
    except Exception:
        return []


def sims(n, nj):
    res = Parallel(n_jobs=nj, verbose=5)(delayed(one)(k) for k in range(n))
    pd.DataFrame([r for rr in res for r in rr]).to_csv("results/obs_fixedband.csv", index=False)


def rec(est, tru):
    e = est - tru
    return dict(n=int(len(e)), r2=float(1 - (e ** 2).sum() / ((tru - tru.mean()) ** 2).sum()),
                spearman=float(pd.Series(np.asarray(est)).corr(pd.Series(np.asarray(tru)), method="spearman")),
                median_bias=float(np.median(e)), mae=float(np.abs(e).mean()))


def summary():
    D = pd.read_csv("results/obs_fixedband.csv").replace([np.inf, -np.inf], np.nan)
    M = joblib.load("results/sbi_sigma250_full.joblib")
    D = D[D[M["features"]].notna().all(1)].reset_index(drop=True)
    pr = predict(M, D)
    D["d_hat"] = fold_delta(pr.th_mu_q50.values, pr.th_c_q50.values)
    D["c_hat"] = pr.th_c_q50.values
    D["d_true"] = fold_delta(D.th_mu.values, D.th_c.values)
    W = D.pivot_table(index="k", values=["d_hat", "c_hat"], columns="version")
    W.columns = [f"{a}_{b}" for a, b in W.columns]
    W = W.dropna().join(D[D.version == "peak"].set_index("k")[["d_true", "th_c", "th_mu"]])
    regions = {"whole prior": np.ones(len(W), bool),
               "near fold (|delta|<2, c>0)": ((W.d_true.abs() < 2) & (W.th_c > 0)).values,
               "healthy region": (W.th_mu.between(-7, -3) & W.th_c.between(1.5, 5.5)).values}
    out = dict(n=int(len(W)))
    for name, m in regions.items():
        w = W[m]
        out[name] = {"peak-centred": rec(w.d_hat_peak, w.d_true), "fixed band": rec(w.d_hat_fixed, w.d_true),
                     "fixed - peak, mean": float((w.d_hat_fixed - w.d_hat_peak).mean()),
                     "fixed - peak, median": float((w.d_hat_fixed - w.d_hat_peak).median()),
                     "spearman fixed vs peak": float(w.d_hat_fixed.corr(w.d_hat_peak, method="spearman"))}
    deep = W[W.d_true < -3]; near = W[W.d_true.between(-1.5, 1.0)]
    out["reach N2 fixed-band estimates"] = {
        k: {"deep (true delta < -3)": [int((deep.d_hat_fixed >= v).sum()), int(len(deep))],
            "near fold (-1.5 to 1)": [int((near.d_hat_fixed >= v).sum()), int(len(near))]} for k, v in N2_FIXED.items()}
    hi = W[W.d_hat_fixed >= min(N2_FIXED.values())]
    out["median true delta with fixed-band estimate >= N2 values"] = float(hi.d_true.median()) if len(hi) else None
    json.dump(out, open("results/obs_fixedband.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "sims":
        sims(int(sys.argv[2]), int(sys.argv[3]))
    else:
        summary()

"""Simulations paired with their own IAAFT linear twin, for a mechanism classifier on twin-differenced statistics.
Classes: intrinsic normal-form dynamics with additive noise (bank prior), intrinsic dynamics with additive plus
state-dependent (multiplicative) noise as in Freyer et al. (2012) ('mult', noise ratio k uniform in 0.25-2.5), and a
noisy oscillator whose Hopf parameter is modulated by a fast Ornstein-Uhlenbeck process (fastmod prior). Spectral
shape is randomised in all classes: the sensor signal is the sum of 1-3 independent sources (shared modulation)
with peak frequencies spaced by up to 0.8 Hz.
The twin is computed on the broadband signal and summarised with the band and noise fraction of the signal,
exactly as for the EEG (ampint.spec.run_power with surrogate_rng).
usage: python scripts/sbi_twinbank.py <config> <intrinsic|mult|fast> <chunk_start> <n_chunks> <n_jobs>
Each chunk = 250 simulations -> results/twinbank_<config>_<class>/chunk_XXXX.npz (resumable)."""
import sys, os, numpy as np, pandas as pd
from joblib import Parallel, delayed
from sbi_bank import CONFIGS, PRIOR, MOD_PRIORS, pink, stable_dt
from ampint.nf import _em_rot, _em_rot_mult, ou_series
from ampint.spec import analytic, noise_fraction, peak_freq, psd, iaaft
from ampint.summ import summaries

OFFSET = {"intrinsic": 50_000_000, "fast": 60_000_000, "mult": 70_000_000}
MULT_K = (0.25, 2.5)
SIZE = 250


def signal(cfg, th, seed, mod, n_src, df_hz, kmult=0.0):
    fs = cfg["fs"]; sigma = th["t0"] ** (-0.75)
    dt = stable_dt(th["mu"], th["c"], th["t0"], mod_sd=0.0 if mod is None else mod["sd"])
    step = max(1, int(np.ceil(1.0 / (fs * dt)))); dt = 1.0 / (fs * step)
    burn = 3.0 + 10 * th["t0"]; n_out = int((th["T"] + 2.0 + burn) * fs)
    rng = np.random.default_rng(seed + 1)
    mus = np.full(n_out, th["mu"] * sigma ** (4 / 3))
    if mod is not None:
        mus = (th["mu"] + mod["sd"] * ou_series(n_out, fs, mod["tau"], rng)) * sigma ** (4 / 3)
    x = 0
    for k in range(n_src):
        if kmult > 0:
            w = _em_rot_mult(mus, np.full(n_out, sigma), float(th["c"] * sigma ** (2 / 3)), float(kmult * sigma ** (-1 / 3)),
                             dt, step, int((seed + 7919 * k) % 2**31))
        else:
            w = _em_rot(mus, np.full(n_out, sigma), float(th["c"] * sigma ** (2 / 3)), dt, step, int((seed + 7919 * k) % 2**31))
        x = x + (w * np.exp(2j * np.pi * (th["f0"] + k * df_hz) * np.arange(n_out) / fs)).real
    x = x[int(burn * fs):]
    return x / (x.std() + 1e-12) + pink(x.size, fs, th["chi"], rng) * 10 ** (-th["snr_db"] / 20)


def one(cfg, cls, seed):
    rng = np.random.default_rng(seed)
    th = dict(mu=rng.uniform(*PRIOR["mu"]), c=rng.uniform(*PRIOR["c"]), t0=10 ** rng.uniform(*PRIOR["log10_t0"]),
              chi=rng.uniform(*PRIOR["chi"]), snr_db=rng.uniform(*PRIOR["snr_db"]), f0=rng.uniform(*cfg["f0"]),
              T=rng.uniform(*cfg["T"]))
    n_src = int(rng.choice([1, 1, 2, 3])); df_hz = float(rng.uniform(0.0, 0.8))
    mod, kmult = None, 0.0
    if cls == "fast":
        mp = MOD_PRIORS["fastmod"]
        mod = dict(sd=rng.uniform(*mp["sd"]), tau=10 ** rng.uniform(*mp["log10_tau"]))
    elif cls == "mult":
        kmult = float(rng.uniform(*MULT_K))
    x = signal(cfg, th, seed, mod, n_src, df_hz, kmult)
    fs = cfg["fs"]
    f, S = psd(x, fs); pk, _ = peak_freq(f, S, cfg["search"])
    lo, hi = pk - cfg["hw"], pk + cfg["hw"]
    rho = noise_fraction(f, S, lo, hi, fs)
    tr = int(fs)
    s = summaries([np.abs(analytic(x, lo, hi, fs)[tr:-tr]) ** 2], rho, fs)
    y = iaaft(x - x.mean(), np.random.default_rng(seed + 999))
    st = summaries([np.abs(analytic(y, lo, hi, fs)[tr:-tr]) ** 2], rho, fs)
    row = {**{"th_" + a: b for a, b in th.items()}, **s, **{"tw_" + a: b for a, b in st.items()},
           "peak": pk, "n_src": n_src, "df_hz": df_hz, "seed": seed}
    if mod:
        row.update(mod_sd=mod["sd"], mod_tau=mod["tau"])
    if kmult:
        row.update(mult_k=kmult)
    return row


def chunk(cfgname, cls, ci):
    out = f"results/twinbank_{cfgname}_{cls}/chunk_{ci:04d}.npz"
    if os.path.exists(out):
        return
    rows = []
    for k in range(SIZE):
        try:
            rows.append(one(CONFIGS[cfgname], cls, OFFSET[cls] + ci * SIZE + k + 1))
        except Exception:
            pass
    df = pd.DataFrame(rows)
    np.savez(out, **{c: df[c].values for c in df.columns})


if __name__ == "__main__":
    cfgname, cls, c0, nc, nj = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    os.makedirs(f"results/twinbank_{cfgname}_{cls}", exist_ok=True)
    Parallel(n_jobs=nj, verbose=5)(delayed(chunk)(cfgname, cls, ci) for ci in range(c0, c0 + nc))

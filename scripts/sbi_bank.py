"""Simulation bank for amortised inference of stochastic-Bautin coordinates from scalp-EEG-like data.
usage: python scripts/sbi_bank.py <config> <chunk_start> <n_chunks> <n_jobs>
Each chunk = 500 simulations -> results/bank_<config>/chunk_XXXX.npz (resumable)."""
import sys, os, numpy as np, pandas as pd
from joblib import Parallel, delayed
from ampint.nf import _em_rot, radial_rate_max, ou_series
from ampint.spec import analytic, noise_fraction, peak_freq, psd
from ampint.summ import summaries, FEATURES

CONFIGS = {
    # alpha rhythm, TDBRAIN-like: 250 Hz, 60-115 s clean, band = peak +- 2 Hz searched in 7-14 Hz
    "alpha": dict(fs=250.0, f0=(8.0, 12.5), search=(7.0, 14.0), hw=2.0, T=(60.0, 115.0)),
    # sleep spindles / sigma, Sleep-EDF-like: 100 Hz
    "sigma": dict(fs=100.0, f0=(11.5, 14.5), search=(10.5, 16.0), hw=1.5, T=(60.0, 300.0)),
    # alpha at 100 Hz (Sleep-EDF wake / N1)
    "alpha100": dict(fs=100.0, f0=(8.0, 12.0), search=(7.0, 13.0), hw=2.0, T=(60.0, 300.0)),
    # sleep spindles (sigma) on 250-Hz-resampled Sleep-EDF, 60-115 s pieces
    "sigma250": dict(fs=250.0, f0=(11.5, 14.5), search=(10.5, 16.0), hw=1.5, T=(60.0, 115.0)),
}
PRIOR = dict(mu=(-8.0, 3.0), c=(-4.0, 8.0), log10_t0=(np.log10(0.02), np.log10(1.0)), chi=(0.5, 2.2), snr_db=(-12.0, 20.0))


def pink(n, fs, chi, rng):
    f = np.fft.rfftfreq(n, 1 / fs); f[0] = f[1]
    X = (rng.standard_normal(f.size) + 1j * rng.standard_normal(f.size)) / f ** (chi / 2)
    x = np.fft.irfft(X, n); return x / x.std()


def stable_dt(mu_t, c_t, t0, f0=None, mod_sd=0.0):
    # accuracy bound: dt <= 0.05 / (max radial Jacobian), amplitude timescale resolved; cap 1 ms
    rate = radial_rate_max(mu_t + 3 * mod_sd, c_t, t0) + 1.0 / t0
    return float(min(1e-3, 0.05 / rate))


def simulate_one(cfg, seed, mod=None):
    rng = np.random.default_rng(seed)
    fs = cfg["fs"]
    th = dict(mu=rng.uniform(*PRIOR["mu"]), c=rng.uniform(*PRIOR["c"]),
              t0=10 ** rng.uniform(*PRIOR["log10_t0"]), chi=rng.uniform(*PRIOR["chi"]),
              snr_db=rng.uniform(*PRIOR["snr_db"]), f0=rng.uniform(*cfg["f0"]), T=rng.uniform(*cfg["T"]))
    return th, observe(cfg, th, seed, mod)


def observe(cfg, th, seed, mod=None, return_power=False):
    """Simulate oscillator with parameters th (+ optional slow modulation) and compute summaries
    (with return_power, also the band power series from which they were computed)."""
    fs = cfg["fs"]
    sigma = th["t0"] ** (-0.75)
    dt = stable_dt(th["mu"], th["c"], th["t0"], mod_sd=0.0 if mod is None else mod["sd"])
    step = max(1, int(np.ceil(1.0 / (fs * dt)))); dt = 1.0 / (fs * step)
    burn = 3.0 + 10 * th["t0"]
    n_out = int((th["T"] + 2.0 + burn) * fs)
    rng = np.random.default_rng(seed + 1)
    mus = np.full(n_out, th["mu"] * sigma ** (4 / 3)); sig = np.full(n_out, sigma)
    if mod is not None:  # hidden slow vigilance-like modulation of mu_t
        o = ou_series(n_out, fs, mod["tau"], rng)
        mus = (th["mu"] + mod["sd"] * o) * sigma ** (4 / 3)
    w = _em_rot(mus, sig, float(th["c"] * sigma ** (2 / 3)), dt, step, int(seed % 2**31))
    z = w * np.exp(2j * np.pi * th["f0"] * np.arange(n_out) / fs)
    z = z[int(burn * fs):]
    x = z.real / (z.real.std() + 1e-12) + pink(z.size, fs, th["chi"], rng) * 10 ** (-th["snr_db"] / 20)
    f, S = psd(x, fs)
    pk, _ = peak_freq(f, S, cfg["search"])
    lo, hi = pk - cfg["hw"], pk + cfg["hw"]
    rho = noise_fraction(f, S, lo, hi, fs)
    a = analytic(x, lo, hi, fs)[int(fs):-int(fs)]
    s = summaries([np.abs(a) ** 2], rho, fs)
    s["peak"] = pk
    return (s, np.abs(a) ** 2) if return_power else s


MOD_PRIORS = {  # hidden modulation of the Hopf parameter: slow (vigilance-like) and fast (extrinsic drive)
    "modbank": dict(sd=(0.25, 2.5), log10_tau=(np.log10(3.0), np.log10(30.0)), seed_offset=10_000_000),
    "fastmod": dict(sd=(0.5, 3.5), log10_tau=(np.log10(0.2), np.log10(5.0)), seed_offset=30_000_000),
}


def chunk(cfgname, ci, size=500, modulated=False):
    """modulated: False (bank), True or 'modbank' (slow modulation), 'fastmod' (fast modulation)."""
    sub = "bank" if not modulated else ("modbank" if modulated is True else modulated)
    out = f"results/{sub}_{cfgname}/chunk_{ci:04d}.npz"
    if os.path.exists(out):
        return
    cfg = CONFIGS[cfgname]
    rows = []
    for k in range(size):
        seed = ci * size + k + 1 + (MOD_PRIORS[sub]["seed_offset"] if modulated else 0)
        try:
            mod = None
            if modulated:
                r = np.random.default_rng(seed + 555); mp = MOD_PRIORS[sub]
                mod = dict(sd=r.uniform(*mp["sd"]), tau=10 ** r.uniform(*mp["log10_tau"]))
            th, s = simulate_one(cfg, seed, mod)
            row = {**{"th_" + a: b for a, b in th.items()}, **s, "seed": seed}
            if mod:
                row.update(mod_sd=mod["sd"], mod_tau=mod["tau"])
            rows.append(row)
        except Exception as e:
            pass
    df = pd.DataFrame(rows)
    np.savez(out, **{c: df[c].values for c in df.columns})


if __name__ == "__main__":
    cfgname, c0, nc, nj = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    modulated = {"mod": "modbank", "fastmod": "fastmod"}.get(sys.argv[5], False) if len(sys.argv) > 5 else False
    os.makedirs(f"results/{modulated or 'bank'}_{cfgname}", exist_ok=True)
    Parallel(n_jobs=nj, verbose=5)(delayed(chunk)(cfgname, ci, 500, modulated) for ci in range(c0, c0 + nc))

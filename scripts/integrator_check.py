"""Accuracy of the Euler-Maruyama integrator used for the simulation banks, benchmarked against the exact stationary
power density of the normal form, p(u) ~ exp(mu u + c u^2/2 - u^3/3).
Representative coordinates (deep focus, focus near the fold, bistable, oscillatory, stiff oscillatory, deep focus with
negative c) are integrated in the co-rotating frame with the initial condition, burn-in and step rule of the banks
(sbi_bank.stable_dt, an integer number of steps per output sample at 250 Hz), and with steps 4 and 16 times smaller,
at amplitude timescales t0 = 0.02 s (the stiffest value of the prior) and 0.1 s.
Compared with the exact density: mean and standard deviation of the scaled power u, probability that power exceeds
five times its mean, Kolmogorov-Smirnov distance; across step sizes also the power-autocorrelation time.
usage: python scripts/integrator_check.py <n_jobs>  -> results/integrator_check.json"""
import sys, json, numpy as np
from joblib import Parallel, delayed
from sbi_bank import stable_dt
from ampint.nf import _em_rot

FS = 250.0
POINTS = {"deep focus": (-5.0, 0.5), "focus near fold": (-2.5, 3.0), "bistable": (-1.0, 3.0),
          "oscillatory": (1.0, 2.0), "stiff oscillatory": (3.0, 8.0), "deep focus, c<0": (-8.0, -4.0)}
T0S = (0.02, 0.1)
FACTORS = (1, 4, 16)
SEEDS = (1, 2, 3, 4)
T_SCALED = 20000.0          # record length in units of t0 (after burn-in)
UG = np.linspace(0, 40, 40001)
np.trapezoid = getattr(np, "trapezoid", None) or np.trapz


def exact(mu, c):
    f = mu * UG + 0.5 * c * UG ** 2 - UG ** 3 / 3
    w = np.exp(f - f.max()); w /= np.trapezoid(w, UG)
    cdf = np.cumsum(w) * (UG[1] - UG[0]); cdf /= cdf[-1]
    m = np.trapezoid(UG * w, UG); sd = np.sqrt(np.trapezoid(UG ** 2 * w, UG) - m * m)
    return dict(mean=m, sd=sd, p5=float(np.trapezoid(w[UG > 5 * m], UG[UG > 5 * m])), cdf=cdf)


def acf_time(u, fs):
    x = u - u.mean(); n = x.size
    F = np.fft.rfft(x, 2 * n); a = np.fft.irfft(F * np.conj(F))[:n]; a /= a[0]
    k = np.argmax(a < np.exp(-1))
    return float(k / fs)


def run(name, t0, fac, seed):
    mu, c = POINTS[name]
    sigma = t0 ** (-0.75)
    dt = stable_dt(mu, c, t0)
    step = max(1, int(np.ceil(1.0 / (FS * dt)))) * fac; dt = 1.0 / (FS * step)
    burn = 3.0 + 10 * t0
    n = int((T_SCALED * t0 + burn) * FS)
    w = _em_rot(np.full(n, mu * sigma ** (4 / 3)), np.full(n, sigma), float(c * sigma ** (2 / 3)), dt, step, seed)
    u = (np.abs(w[int(burn * FS):]) ** 2) * sigma ** (-2 / 3)
    return dict(name=name, t0=t0, fac=fac, seed=seed, dt=dt, u=u)


def summarize(rows):
    out = []
    for name in POINTS:
        mu, c = POINTS[name]; ex = exact(mu, c)
        for t0 in T0S:
            for fac in FACTORS:
                rr = [r for r in rows if r["name"] == name and r["t0"] == t0 and r["fac"] == fac]
                u = np.concatenate([r["u"] for r in rr])
                m, sd = float(u.mean()), float(u.std())
                ecdf = np.searchsorted(np.sort(u), UG, side="right") / u.size
                out.append(dict(point=name, mu=mu, c=c, t0=t0, step_factor=fac, dt_s=rr[0]["dt"],
                                mean=m, mean_exact=ex["mean"], rel_err_mean=m / ex["mean"] - 1,
                                sd=sd, sd_exact=ex["sd"], rel_err_sd=sd / ex["sd"] - 1,
                                p5=float((u > 5 * ex["mean"]).mean()), p5_exact=ex["p5"],
                                ks=float(np.max(np.abs(ecdf - ex["cdf"]))),
                                acf_time_s=float(np.mean([acf_time(r["u"], FS) for r in rr]))))
    return out


if __name__ == "__main__":
    jobs = [(nm, t0, f, s) for nm in POINTS for t0 in T0S for f in FACTORS for s in SEEDS]
    rows = Parallel(n_jobs=int(sys.argv[1]), verbose=5)(delayed(run)(*j) for j in jobs)
    out = summarize(rows)
    # Monte Carlo reference: KS distance between two independent halves of the seeds at the bank step
    for o in out:
        rr = [r for r in rows if r["name"] == o["point"] and r["t0"] == o["t0"] and r["fac"] == o["step_factor"]]
        a = np.sort(np.concatenate([r["u"] for r in rr[:2]])); b = np.sort(np.concatenate([r["u"] for r in rr[2:]]))
        g = np.linspace(0, max(a[-1], b[-1]), 4000)
        o["ks_between_halves"] = float(np.max(np.abs(np.searchsorted(a, g, "right") / a.size - np.searchsorted(b, g, "right") / b.size)))
    json.dump(out, open("results/integrator_check.json", "w"), indent=1)
    for o in out:
        print(f"{o['point']:20s} t0={o['t0']:.2f} x{o['step_factor']:2d} dt={o['dt_s']:.2e} mean {o['rel_err_mean']:+.4f} "
              f"sd {o['rel_err_sd']:+.4f} KS {o['ks']:.4f} (halves {o['ks_between_halves']:.4f}) p5 {o['p5']:.5f}/{o['p5_exact']:.5f} acf {o['acf_time_s']:.4f}")

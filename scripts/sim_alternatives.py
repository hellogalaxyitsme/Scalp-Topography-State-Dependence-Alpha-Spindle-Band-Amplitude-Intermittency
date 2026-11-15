"""Simulation checks for two alternative explanations of near-fold estimates.
(1) Fast extrinsic drive: a deep noisy focus whose Hopf parameter is modulated by an Ornstein-Uhlenbeck process
    with correlation time 0.3-3 s (comparable to slow-oscillation-triggered spindles), sigma and alpha settings.
(2) Source mixing at the sensor: the sum of two or three independent oscillators with identical coordinates.
Each simulation is passed through the observation pipeline of the simulation bank and the trained posterior.
usage: python scripts/sim_alternatives.py <n_jobs>"""
import sys, json, joblib, numpy as np, pandas as pd
from joblib import Parallel, delayed
from sbi_bank import CONFIGS, observe, pink, stable_dt
from sbi_fit import predict
from ampint.nf import _em_rot, fold_delta
from ampint.spec import psd, peak_freq, noise_fraction, analytic
from ampint.summ import summaries


def drive_case(cfgname, i, sd, tau):
    rng = np.random.default_rng(5_000_000 + i)
    cfg = CONFIGS[cfgname]
    th = dict(mu=rng.uniform(-7, -4), c=rng.uniform(-1, 2), t0=10 ** rng.uniform(-1.3, -0.5), chi=rng.uniform(0.8, 1.6),
              snr_db=rng.uniform(0, 10), f0=rng.uniform(*cfg["f0"]), T=100.0)
    s = observe(cfg, th, 6_000_000 + i, None if sd == 0 else dict(sd=sd, tau=tau))
    return dict(case="drive", cfg=cfgname, sd=sd, tau=tau, true_delta=float(fold_delta(th["mu"], th["c"])), **s)


def mixed(cfg, th, seed, n_src, df_hz):
    """As observe(), but the scalp signal is the sum of n_src independent oscillators (peak frequencies spaced by df_hz)."""
    fs = cfg["fs"]; sigma = th["t0"] ** (-0.75)
    dt = stable_dt(th["mu"], th["c"], th["t0"]); step = max(1, int(np.ceil(1.0 / (fs * dt)))); dt = 1.0 / (fs * step)
    burn = 3.0 + 10 * th["t0"]; n_out = int((th["T"] + 2.0 + burn) * fs)
    rng = np.random.default_rng(seed + 1)
    x = 0
    for k in range(n_src):
        w = _em_rot(np.full(n_out, th["mu"] * sigma ** (4 / 3)), np.full(n_out, sigma), float(th["c"] * sigma ** (2 / 3)),
                    dt, step, int((seed + 7919 * k) % 2**31))
        x = x + (w * np.exp(2j * np.pi * (th["f0"] + k * df_hz) * np.arange(n_out) / fs)).real
    x = x[int(burn * fs):]
    x = x / x.std() + pink(x.size, fs, th["chi"], rng) * 10 ** (-th["snr_db"] / 20)
    f, S = psd(x, fs); pk, _ = peak_freq(f, S, cfg["search"])
    lo, hi = pk - cfg["hw"], pk + cfg["hw"]
    a = analytic(x, lo, hi, fs)[int(fs):-int(fs)]
    s = summaries([np.abs(a) ** 2], noise_fraction(f, S, lo, hi, fs), fs); s["peak"] = pk
    return s


def mix_case(i, d_true, n_src, df_hz):
    rng = np.random.default_rng(7_000_000 + i)
    c = 4.0
    th = dict(mu=d_true - c * c / 4, c=c, t0=0.3, chi=1.2, snr_db=5.0, f0=10.0, T=100.0)
    s = mixed(CONFIGS["alpha"], th, 8_000_000 + i, n_src, df_hz)
    return dict(case="mix", cfg="alpha", n_src=n_src, df_hz=df_hz, true_delta=d_true, **s)


if __name__ == "__main__":
    nj = int(sys.argv[1])
    jobs = []
    for cfgname in ("sigma250", "alpha"):
        for sd, tau in [(0, 0)] + [(sd, tau) for sd in (1.0, 2.0, 3.0) for tau in (0.3, 1.0, 3.0)]:
            jobs += [delayed(drive_case)(cfgname, len(jobs), sd, tau) for _ in range(40)]
    for d_true in (-3.0, -1.5, 0.0):
        for n_src, df_hz in ((1, 0.0), (2, 0.0), (2, 0.5), (3, 0.4)):
            jobs += [delayed(mix_case)(len(jobs) + k, d_true, n_src, df_hz) for k in range(30)]
    rows = Parallel(n_jobs=nj, verbose=2)(jobs)
    df = pd.DataFrame(rows)
    out = []
    for cfgname, model in (("sigma250", "results/sbi_sigma250_full.joblib"), ("alpha", "results/sbi_alpha_full.joblib")):
        d = df[df.cfg == cfgname].reset_index(drop=True)
        pr = predict(joblib.load(model), d)
        out.append(pd.concat([d, pr.reset_index(drop=True)], axis=1))
    R = pd.concat(out, ignore_index=True)
    R["delta_hat"] = fold_delta(R.th_mu_q50, R.th_c_q50)
    R.to_csv("results/sim_alternatives.csv", index=False)
    pd.set_option("display.width", 220)
    g1 = R[R.case == "drive"].groupby(["cfg", "sd", "tau"]).agg(n=("delta_hat", "size"), true_delta=("true_delta", "median"),
                                                               delta_hat=("delta_hat", "median"), p_bist=("p_bistable", "mean"),
                                                               flag=("slow_flag", "mean"))
    g2 = R[R.case == "mix"].groupby(["true_delta", "n_src", "df_hz"]).agg(n=("delta_hat", "size"), delta_hat=("delta_hat", "median"),
                                                                         p_bist=("p_bistable", "mean"), flag=("slow_flag", "mean"))
    print(g1.round(3).to_string()); print(g2.round(3).to_string())
    json.dump(dict(drive=g1.reset_index().to_dict("records"), mixing=g2.reset_index().to_dict("records")),
              open("results/sim_alternatives.json", "w"), indent=1)

"""Sensitivity of the simulations to their initial condition.
The banks start every realization at w = 0.1 (units of z) and discard 3 s + 10 t0. In slow-switching parts of the
prior (bistable coordinates with long t0) the lower basin may not be left within this burn-in. Here each simulation is
run twice with identical noise: from the bank initialization, and from the exact stationary density of Eq. (2)
(u drawn from p(u | mu, c) at the initial value of mu, uniform phase, z0 = sigma^(1/3) sqrt(u) e^(i theta)), with
the same burn-in. Both versions pass through the observation pipeline, the alpha posterior and the main classifier.
  spot     - raw oscillator at (mu, c, t0) = (-8, 7, 1 s): mean scaled power and lower-basin occupancy over 60 s
  sims     - paired simulations from the alpha bank priors (constant parameters) and the modulated prior
  summary  - coordinate and score differences by regime, recovery of the true fold coordinate for both versions
usage: python scripts/init_check.py spot <n_jobs> | sims <n_const> <n_mod> <n_jobs> | summary
       -> results/init_check_spot.json, results/init_check.csv, results/init_check.json"""
import sys, json, joblib, numpy as np, pandas as pd
from numba import njit
from joblib import Parallel, delayed
from sbi_bank import CONFIGS, PRIOR, MOD_PRIORS, pink, stable_dt
from sbi_fit import predict
from mech_twin import DIFF, NEAR
from ampint.nf import ou_series, fold_delta
from ampint.spec import analytic, noise_fraction, peak_freq, psd, iaaft
from ampint.summ import summaries

CFG = CONFIGS["alpha"]
OFFSET = 96_000_000


@njit(cache=True)
def em_rot_init(mus, sigmas, c, dt, step, seed, w0r, w0i):
    """Same as ampint.nf._em_rot (identical noise stream for a given seed) with a chosen initial condition."""
    np.random.seed(seed)
    wr, wi = w0r, w0i
    m = mus.size
    outr = np.empty(m); outi = np.empty(m)
    for k in range(m):
        mu = mus[k]; sq = sigmas[k] * np.sqrt(dt)
        for j in range(step):
            r2 = wr * wr + wi * wi
            g = mu + c * r2 - r2 * r2
            wr += g * wr * dt + sq * np.random.randn()
            wi += g * wi * dt + sq * np.random.randn()
        outr[k] = wr; outi[k] = wi
    return outr + 1j * outi


def stationary_u(mu, c, rng):
    u = np.linspace(0, 60, 120001)
    f = mu * u + 0.5 * c * u * u - u ** 3 / 3
    keep = f > f.max() - 40
    u = u[: np.nonzero(keep)[0].max() + 2]; f = f[: u.size]
    w = np.exp(f - f.max()); cdf = np.cumsum(w); cdf /= cdf[-1]
    return float(np.interp(rng.random(), cdf, u))


def exact_moments(mu, c):
    u = np.linspace(0, 60, 120001)
    f = mu * u + 0.5 * c * u * u - u ** 3 / 3
    w = np.exp(f - f.max()); w /= w.sum()
    return float((u * w).sum())


def run(th, seed, mod, init, return_w=False):
    fs = CFG["fs"]; sigma = th["t0"] ** (-0.75)
    dt = stable_dt(th["mu"], th["c"], th["t0"], mod_sd=0.0 if mod is None else mod["sd"])
    step = max(1, int(np.ceil(1.0 / (fs * dt)))); dt = 1.0 / (fs * step)
    burn = 3.0 + 10 * th["t0"]
    n_out = int((th["T"] + 2.0 + burn) * fs)
    rng = np.random.default_rng(seed + 1)
    mt = np.full(n_out, th["mu"])
    if mod is not None:
        mt = th["mu"] + mod["sd"] * ou_series(n_out, fs, mod["tau"], rng)
    mus = mt * sigma ** (4 / 3); sig = np.full(n_out, sigma)
    if init == "bank":
        w0 = 0.1 + 0j
    else:
        r0 = np.random.default_rng(seed + 77)
        w0 = sigma ** (1 / 3) * np.sqrt(stationary_u(mt[0], th["c"], r0)) * np.exp(2j * np.pi * r0.random())
    w = em_rot_init(mus, sig, float(th["c"] * sigma ** (2 / 3)), dt, step, int(seed % 2**31), float(w0.real), float(w0.imag))
    if return_w:
        return w[int(burn * fs):] , sigma
    u_mean = float(np.mean(np.abs(w[int(burn * fs):]) ** 2) * sigma ** (-2 / 3))
    z = (w * np.exp(2j * np.pi * th["f0"] * np.arange(n_out) / fs))[int(burn * fs):]
    x = z.real / (z.real.std() + 1e-12) + pink(z.size, fs, th["chi"], rng) * 10 ** (-th["snr_db"] / 20)
    f, S = psd(x, fs)
    pk, _ = peak_freq(f, S, CFG["search"])
    lo, hi = pk - CFG["hw"], pk + CFG["hw"]
    rho = noise_fraction(f, S, lo, hi, fs)
    tr = int(fs)
    s = summaries([np.abs(analytic(x, lo, hi, fs)[tr:-tr]) ** 2], rho, fs)
    y = iaaft(x - x.mean(), np.random.default_rng(seed + 999))
    st = summaries([np.abs(analytic(y, lo, hi, fs)[tr:-tr]) ** 2], rho, fs)
    return dict(s, peak=pk, u_mean=u_mean, **{"tw_" + k: v for k, v in st.items()})


def one(k, cls):
    seed = OFFSET + (0 if cls == "const" else 1_000_000) + k
    rng = np.random.default_rng(seed)
    th = dict(mu=rng.uniform(*PRIOR["mu"]), c=rng.uniform(*PRIOR["c"]), t0=10 ** rng.uniform(*PRIOR["log10_t0"]),
              chi=rng.uniform(*PRIOR["chi"]), snr_db=rng.uniform(*PRIOR["snr_db"]), f0=rng.uniform(*CFG["f0"]),
              T=rng.uniform(*CFG["T"]))
    mod = None
    if cls == "mod":
        r = np.random.default_rng(seed + 555); mp = MOD_PRIORS["fastmod"]
        mod = dict(sd=r.uniform(*mp["sd"]), tau=10 ** r.uniform(*mp["log10_tau"]))
    rows = []
    try:
        for init in ("bank", "stat"):
            rows.append(dict(k=k, cls=cls, init=init, seed=seed, **{"th_" + a: b for a, b in th.items()},
                             mod_sd=mod["sd"] if mod else 0.0, mod_tau=mod["tau"] if mod else np.nan,
                             u_exact=exact_moments(th["mu"], th["c"]), **run(th, seed, mod, init)))
    except Exception:
        return []
    return rows


def spot():
    th = dict(mu=-8.0, c=7.0, t0=1.0, chi=1.0, snr_db=0.0, f0=10.0, T=60.0)
    ex = exact_moments(-8.0, 7.0); um = (7 - np.sqrt(17)) / 2
    u = np.linspace(0, 60, 120001); f = -8 * u + 3.5 * u * u - u ** 3 / 3; w = np.exp(f - f.max()); w /= w.sum()
    occ_ex = float(w[u < um].sum())
    out = {}
    for init in ("bank", "stat"):
        res = Parallel(n_jobs=int(sys.argv[2]) if len(sys.argv) > 2 else 8)(
            delayed(run)(th, OFFSET + 5_000_000 + i, None, init, True) for i in range(300))
        m = [float(np.mean(np.abs(w_) ** 2) * s_ ** (-2 / 3)) for w_, s_ in res]
        o = [float(np.mean(np.abs(w_) ** 2 * s_ ** (-2 / 3) < um)) for w_, s_ in res]
        first = [float(np.abs(w_[0]) ** 2 * s_ ** (-2 / 3)) for w_, s_ in res]
        out[init] = dict(mean_u=float(np.mean(m)), se=float(np.std(m) / np.sqrt(len(m))), lower_occupancy=float(np.mean(o)),
                         lower_occupancy_se=float(np.std(o) / np.sqrt(len(o))), u_at_end_of_burnin=float(np.mean(first)))
    out["exact"] = dict(mean_u=ex, lower_occupancy=occ_ex, u_minus=float(um))
    json.dump(out, open("results/init_check_spot.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


def sims(nc, nm, nj):
    jobs = [(k, "const") for k in range(nc)] + [(k, "mod") for k in range(nm)]
    res = Parallel(n_jobs=nj, verbose=5)(delayed(one)(k, c) for k, c in jobs)
    pd.DataFrame([r for rr in res for r in rr]).to_csv("results/init_check.csv", index=False)


def rec(est, tru):
    e = est - tru
    return dict(n=int(len(e)), r2=float(1 - (e ** 2).sum() / ((tru - tru.mean()) ** 2).sum()) if len(e) > 2 else None,
                spearman=float(pd.Series(np.asarray(est)).corr(pd.Series(np.asarray(tru)), method="spearman")) if len(e) > 2 else None,
                median_bias=float(np.median(e)) if len(e) else None, mae=float(np.abs(e).mean()) if len(e) else None)


def summary():
    D = pd.read_csv("results/init_check.csv").replace([np.inf, -np.inf], np.nan)
    M = joblib.load("results/sbi_alpha_full.joblib")
    D = D[D[M["features"]].notna().all(1)].reset_index(drop=True)
    pr = predict(M, D)
    D["d_hat"] = fold_delta(pr.th_mu_q50.values, pr.th_c_q50.values)
    D["mu_hat"], D["c_hat"], D["t0_hat"] = pr.th_mu_q50.values, pr.th_c_q50.values, pr.log_t0_q50.values
    C = joblib.load("results/mechtwb_alpha.joblib")
    for f in DIFF:
        D["d_" + f] = D[f] - D["tw_" + f]
    ok = D[C["cols"]].notna().all(1); D["score"] = np.nan
    D.loc[ok, "score"] = C["iso"].predict(C["clf"].predict_proba(D.loc[ok, C["cols"]].values)[:, 1])
    D["u_ratio"] = D.u_mean / D.u_exact
    W = D.pivot_table(index=["k", "cls"], columns="init", values=["d_hat", "mu_hat", "c_hat", "t0_hat", "score", "u_ratio"])
    W.columns = [f"{a}_{b}" for a, b in W.columns]
    W = W.dropna().reset_index()
    th = D[D.init == "bank"][["k", "cls", "th_mu", "th_c", "th_t0"]]
    W = W.merge(th, on=["k", "cls"])
    W["d_true"] = fold_delta(W.th_mu.values, W.th_c.values)
    W["dd"] = W.d_hat_stat - W.d_hat_bank
    reg = {"all": np.ones(len(W), bool), "focus (delta<0)": W.d_true < 0,
           "bistable (delta>0, mu<0)": (W.d_true > 0) & (W.th_mu < 0), "oscillatory (mu>0)": W.th_mu > 0,
           "bistable, t0>0.3 s": (W.d_true > 0) & (W.th_mu < 0) & (W.th_t0 > 0.3),
           "near fold (|delta|<2, c>0)": (W.d_true.abs() < 2) & (W.th_c > 0),
           "healthy box": W.th_mu.between(-7, -3) & W.th_c.between(1.5, 5.5)}
    out = dict(n=int(len(W)))
    for cls in ("const", "mod"):
        r = {}
        for name, m in reg.items():
            w = W[np.asarray(m) & (W.cls == cls).values]
            if len(w) < 5:
                continue
            r[name] = dict(n=int(len(w)),
                           mean_d_hat_diff=float(w.dd.mean()), mean_abs_d_hat_diff=float(w.dd.abs().mean()),
                           frac_abs_d_hat_diff_gt_0p5=float((w.dd.abs() > 0.5).mean()),
                           mean_c_hat_diff=float((w.c_hat_stat - w.c_hat_bank).mean()),
                           mean_score_diff=float((w.score_stat - w.score_bank).mean()),
                           median_u_ratio_bank=float(w.u_ratio_bank.median()), median_u_ratio_stat=float(w.u_ratio_stat.median()),
                           frac_u_ratio_bank_off_5pct=float(((w.u_ratio_bank - 1).abs() > 0.05).mean()),
                           frac_u_ratio_stat_off_5pct=float(((w.u_ratio_stat - 1).abs() > 0.05).mean()),
                           recovery_bank=rec(w.d_hat_bank, w.d_true), recovery_stat=rec(w.d_hat_stat, w.d_true))
        out[cls] = r
    aff = W[(W.cls == "const") & (W.dd.abs() > 0.5)]
    out["affected (const, |diff|>0.5)"] = dict(n=int(len(aff)), frac=float(len(aff) / max(1, (W.cls == "const").sum())),
                                              median_t0=float(aff.th_t0.median()) if len(aff) else None,
                                              frac_bistable=float(((aff.d_true > 0) & (aff.th_mu < 0)).mean()) if len(aff) else None,
                                              median_mu=float(aff.th_mu.median()) if len(aff) else None,
                                              median_c=float(aff.th_c.median()) if len(aff) else None)
    W.to_csv("results/init_check_paired.csv", index=False)
    json.dump(out, open("results/init_check.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    a = sys.argv[1]
    if a == "spot":
        spot()
    elif a == "sims":
        sims(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    else:
        summary()

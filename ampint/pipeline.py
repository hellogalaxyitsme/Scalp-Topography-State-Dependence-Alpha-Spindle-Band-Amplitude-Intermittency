"""Per-channel normal-form analysis shared by all datasets."""
import numpy as np
from scipy.signal import welch
from .spec import clean_runs, run_power, noise_fraction, aperiodic_fit, peak_freq
from .nf import fit_power, fit_power_deconv, envelope_cv
from .fp import adequacy, moments_u


def clean_psd(x, mask, fs, min_len_s=8.0):
    nper = int(4 * fs)
    acc, n = None, 0
    for a, b in clean_runs(mask, max(int(min_len_s * fs), nper)):
        f, S = welch(x[a:b] - x[a:b].mean(), fs, nperseg=nper, noverlap=nper // 2)
        w = (b - a)
        acc = S * w if acc is None else acc + S * w; n += w
    if acc is None:
        return None, None
    return f, acc / n


def osc_scale(d, rho, Q):
    """Var(P_osc)/Var(P_obs) in mean-normalised units (long-lag ACF scaling under additive background)."""
    m_u, v_u = moments_u(d["mu_t"], d["c_t"])
    q = Q / Q.mean()
    return float(min(1.0, v_u * ((1 - rho) / m_u) ** 2 / q.var()))


def nf_stats(runs, rho, fs, with_adequacy=True):
    Q = np.concatenate(runs)
    d = fit_power_deconv(Q, rho)
    out = dict(mu=d["mu_t"], c=d["c_t"], regime=d["regime"], cv=envelope_cv(Q), n=Q.size,
               a1=d["a1"], a2=d["a2"], a3=d["a3"])
    nv = fit_power(Q)
    out.update(mu_naive=nv["mu_t"], c_naive=nv["c_t"])
    if with_adequacy:
        sc = osc_scale(d, rho, Q)
        ad = adequacy(runs, fs, d["mu_t"], d["c_t"], scale=sc)
        out.update(t0=ad["t0"], excess=ad["slow_excess"], tail_emp=ad["tail_emp"], tail_pred=ad["tail_pred"], scale=sc)
    return out


def analyze_channel(x, mask, fs, lo, hi, n_surr=5, seed=0, min_len_s=8.0):
    """Normal-form coordinates of the [lo,hi] rhythm + IAAFT-surrogate null for one channel."""
    f, S = clean_psd(x, mask, fs, min_len_s)
    if f is None:
        return None
    rho = noise_fraction(f, S, lo, hi, fs)
    runs = run_power(x, mask, lo, hi, fs, min_len_s)
    if sum(r.size for r in runs) < 20 * fs:
        return None
    res = dict(rho=rho, clean_s=sum(r.size for r in runs) / fs, nruns=len(runs))
    res.update(nf_stats(runs, rho, fs))
    rng = np.random.default_rng(seed)
    sur = []
    for k in range(n_surr):
        sr = run_power(x, mask, lo, hi, fs, min_len_s, surrogate_rng=rng)
        sur.append(nf_stats(sr, rho, fs))
    for key in ["mu", "c", "cv", "excess", "mu_naive", "c_naive"]:
        v = np.array([s[key] for s in sur], float)
        res[key + "_sm"] = float(np.nanmean(v)); res[key + "_ss"] = float(np.nanstd(v))
    return res

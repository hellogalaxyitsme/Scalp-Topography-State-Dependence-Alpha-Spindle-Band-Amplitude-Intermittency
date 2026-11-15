"""Summary statistics shared by simulations and data (must be computed identically)."""
import numpy as np
from .nf import fit_power, fit_power_deconv
from .fp import empirical_acf

ACF_LAGS = np.array([0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0])
TAIL = (2.0, 10.0)
QS = [0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
CLIP = 30.0


def summaries(runs, rho, fs):
    """runs: list of band-power arrays (clean runs). Returns dict of summaries (+ long-lag tail for PPC)."""
    Q = np.concatenate(runs)
    q = Q / Q.mean()
    d = fit_power_deconv(Q, rho)
    n = fit_power(Q)
    amp = np.sqrt(q)
    ac = empirical_acf(runs, fs, TAIL[1])
    lags = np.arange(ac.size) / fs
    s = dict(rho=rho, T=Q.size / fs,
             mu_d=np.clip(d["mu_t"], -CLIP, CLIP), c_d=np.clip(d["c_t"], -CLIP, CLIP),
             mu_n=np.clip(n["mu_t"], -CLIP, CLIP), c_n=np.clip(n["c_t"], -CLIP, CLIP),
             cv=amp.std() / amp.mean(), kurt=float(((q - 1) ** 4).mean() / q.var() ** 2))
    for p, v in zip(QS, np.quantile(q, QS)):
        s[f"lq{int(p * 100)}"] = float(np.log(v + 1e-6))
    for L in ACF_LAGS:
        s[f"acf{L:g}"] = float(np.interp(L, lags, ac))
    # richer stored descriptors: log-power histogram (fixed bins) and dense ACF
    h, _ = np.histogram(np.log(q + 1e-9), HBINS)
    h = h / max(1, h.sum())
    for i, v in enumerate(h):
        s[f"h{i:02d}"] = float(v)
    for i, L in enumerate(DLAGS):
        s[f"a{i:02d}"] = float(np.interp(L, lags, ac))
    m = (lags >= TAIL[0]) & (lags <= TAIL[1])
    s["tail"] = float(np.trapz(ac[m], lags[m]))
    return s


HBINS = np.r_[-np.inf, np.linspace(-6, 2.5, 29), np.inf]       # 30 bins of log(P/mean P)
DLAGS = np.geomspace(0.02, 3.0, 40)                              # 40 lags (s)
FEATURES = ["rho", "T", "mu_d", "c_d", "mu_n", "c_n", "cv", "kurt"] + [f"lq{int(p * 100)}" for p in QS] + \
           [f"acf{L:g}" for L in ACF_LAGS] + [f"h{i:02d}" for i in range(30)] + [f"a{i:02d}" for i in range(40)]

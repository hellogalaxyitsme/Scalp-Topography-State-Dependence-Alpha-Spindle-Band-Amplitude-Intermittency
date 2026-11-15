"""1-D Fokker-Planck dynamics of oscillation power under the stochastic Bautin normal form.

Dimensionless power u = P / sigma^(2/3), time tau = t * sigma^(4/3) =: t / t0 obey (Ito)
    du = 2 (mu_t u + c_t u^2 - u^3 + 1) dtau + 2 sqrt(u) dW,
whose stationary density is p(u) ~ exp(mu_t u + c_t u^2/2 - u^3/3) (same family as nf.py).
Given (mu_t, c_t) the only free dynamical constant is t0 (seconds). The generator is discretised as
a reversible birth-death chain (upwind), giving the exact-in-the-limit autocorrelation of u.
"""
import numpy as np
from scipy.linalg import eigh


def _grid(mu_t, c_t, n=300):
    # support: up to where log-density has dropped by 40 below its max
    u = np.linspace(0, 60, 20001)
    e = mu_t * u + c_t * u ** 2 / 2 - u ** 3 / 3
    keep = e > e.max() - 40
    umax = u[np.where(keep)[0].max()] * 1.1 + 1e-3
    return np.linspace(0, umax, n)


def generator(mu_t, c_t, n=300):
    """Returns grid u, stationary pi (from detailed balance) and symmetric matrix S with
    L = D^(-1/2) S D^(1/2) (D = diag(pi)); eigenpairs of S give relaxation spectrum."""
    u = _grid(mu_t, c_t, n)
    h = u[1] - u[0]
    a = 2 * (mu_t * u + c_t * u ** 2 - u ** 3 + 1)   # drift
    b = 2 * u                                         # diffusion coefficient D2/2 = 4u/2
    up = b / h ** 2 + np.maximum(a, 0) / h
    dn = b / h ** 2 + np.maximum(-a, 0) / h
    up[-1] = 0.0; dn[0] = 0.0
    # stationary from detailed balance: pi_{i+1}/pi_i = up_i / dn_{i+1}
    lr = np.zeros(n)
    lr[1:] = np.cumsum(np.log(up[:-1] + 1e-300) - np.log(dn[1:] + 1e-300))
    pi = np.exp(lr - lr.max()); pi /= pi.sum()
    # symmetric form: S_{i,i+1} = sqrt(up_i * dn_{i+1})
    off = np.sqrt(up[:-1] * dn[1:])
    S = np.diag(-(up + dn)) + np.diag(off, 1) + np.diag(off, -1)
    return u, pi, S


def acf_u(mu_t, c_t, lags_tau, n=300):
    """Normalised autocorrelation of u at dimensionless lags."""
    u, pi, S = generator(mu_t, c_t, n)
    lam, V = eigh(S)
    f = u - (pi * u).sum()
    w = (V.T @ (np.sqrt(pi) * f))           # projections
    var = (pi * f * f).sum()
    lags_tau = np.atleast_1d(lags_tau)
    C = (np.exp(np.outer(lags_tau, lam)) * w ** 2).sum(1) / var
    return C


def moments_u(mu_t, c_t, n=300):
    u, pi, _ = generator(mu_t, c_t, n)
    m = (pi * u).sum(); v = (pi * (u - m) ** 2).sum()
    return m, v


def fit_t0(emp_lags_s, emp_acf, mu_t, c_t, scale=1.0, t0_grid=None):
    """Least-squares t0 so that scale * C_u(lag/t0) matches the empirical ACF on the given lags.
    `scale` = Var(P_osc)/Var(P_obs) accounts for additive in-band background."""
    t0_grid = t0_grid if t0_grid is not None else np.exp(np.linspace(np.log(0.005), np.log(5.0), 120))
    best = (np.inf, None)
    for t0 in t0_grid:
        pred = scale * acf_u(mu_t, c_t, emp_lags_s / t0)
        err = np.mean((pred - emp_acf) ** 2)
        if err < best[0]:
            best = (err, t0)
    return best[1], best[0]


def empirical_acf(P, fs, max_lag_s):
    """Autocorrelation of power; P may be one array or a list of clean runs (pooled, global mean)."""
    runs = P if isinstance(P, (list, tuple)) else [P]
    mu = np.concatenate(runs).mean()
    nl = int(max_lag_s * fs)
    num = np.zeros(nl + 1); den = np.zeros(nl + 1)
    for r in runs:
        p = r - mu
        L = min(nl, p.size - 1)
        nfft = 1 << int(np.ceil(np.log2(2 * p.size)))
        F = np.fft.rfft(p, nfft)
        ac = np.fft.irfft(F * np.conj(F), nfft)[:L + 1]
        num[:L + 1] += ac; den[:L + 1] += np.arange(p.size, p.size - L - 1, -1)
    ac = num / np.maximum(den, 1)
    return ac / ac[0]


def adequacy(P, fs, mu_t, c_t, scale=1.0, fit_lags=(0.3, 1.0), test_lags=(2.0, 10.0)):
    """Fit t0 on short/intermediate lags, then measure long-lag ACF excess (data - model), in s.
    ~0 for a 1-D Markov Bautin process; >0 when an unmodelled slow variable modulates power."""
    ac = empirical_acf(P, fs, test_lags[1])
    lags = np.arange(ac.size) / fs
    mfit = (lags >= fit_lags[0]) & (lags <= fit_lags[1])
    t0, err = fit_t0(lags[mfit][::5], ac[mfit][::5], mu_t, c_t, scale)
    mt = (lags >= test_lags[0]) & (lags <= test_lags[1])
    pred = scale * acf_u(mu_t, c_t, lags[mt][::5] / t0)
    excess = np.trapz(ac[mt][::5] - pred, lags[mt][::5])
    return dict(t0=t0, fit_err=err, slow_excess=float(excess),
                tail_emp=float(np.trapz(ac[mt][::5], lags[mt][::5])), tail_pred=float(np.trapz(pred, lags[mt][::5])))

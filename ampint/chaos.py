"""Classical chaos / complexity indices with surrogate control.

All indices are computed on the same clean broadband segment and on IAAFT surrogates that preserve its
power spectrum and amplitude distribution exactly (linear-Gaussian-rescaled null, Schreiber & Schmitz 1996).
Excess over the surrogate mean (z-units) isolates the part of each index not explained by linear structure.
"""
import numpy as np
from scipy.spatial import cKDTree
import antropy as ant
from .spec import iaaft


def embed(x, m, tau):
    n = x.size - (m - 1) * tau
    return np.stack([x[i * tau:i * tau + n] for i in range(m)], 1)


def lle_rosenstein(x, fs, m=10, tau=3, theiler_s=0.1, fit_s=(0.0, 0.15), horizon_s=0.3):
    """Largest Lyapunov exponent (1/s) via Rosenstein et al. (1993) mean log divergence."""
    Y = embed(x, m, tau)
    n = Y.shape[0]
    th = int(theiler_s * fs); H = int(horizon_s * fs)
    tree = cKDTree(Y)
    k = th * 2 + 5
    d, idx = tree.query(Y[:n - H], k=min(k, n - 1))
    nn = np.full(n - H, -1)
    for i in range(n - H):
        for dj, j in zip(d[i], idx[i]):
            if abs(j - i) > th and j < n - H and dj > 0:
                nn[i] = j; break
    ok = nn >= 0
    ii = np.where(ok)[0]; jj = nn[ok]
    div = np.empty(H)
    for h in range(H):
        dd = np.linalg.norm(Y[ii + h] - Y[jj + h], axis=1)
        div[h] = np.mean(np.log(dd[dd > 0]))
    t = np.arange(H) / fs
    sel = (t >= fit_s[0]) & (t <= fit_s[1])
    return float(np.polyfit(t[sel], div[sel], 1)[0])


def zero_one_K(x, n_c=100, rng=None, downsample=None):
    """0-1 test for chaos, correlation method (Gottwald & Melbourne 2009). K ~ 0 regular, ~1 chaotic.
    Median over random c in (pi/5, 4pi/5). Oversampled flows are decimated to ~3 samples per mean
    cycle (mean cycle from zero-crossing rate), as required for continuous-time signals."""
    rng = rng or np.random.default_rng(0)
    if downsample is None:
        xc = x - x.mean()
        ncyc = max(1.0, np.sum(np.diff(np.sign(xc)) != 0) / 2.0)
        downsample = max(1, int(round(x.size / ncyc / 3.0)))
    phi = x[::downsample]
    phi = (phi - phi.mean()) / phi.std()
    N = phi.size; ncut = N // 10
    j = np.arange(1, N + 1)
    Ks = []
    for c in rng.uniform(np.pi / 5, 4 * np.pi / 5, n_c):
        p = np.cumsum(phi * np.cos(j * c)); q = np.cumsum(phi * np.sin(j * c))
        n = np.arange(1, ncut + 1)
        M = np.array([np.mean((p[k:] - p[:-k]) ** 2 + (q[k:] - q[:-k]) ** 2) for k in n])
        Vosc = (np.mean(phi) ** 2) * (1 - np.cos(n * c)) / (1 - np.cos(c))
        D = M - Vosc
        Ks.append(np.corrcoef(n, D)[0, 1])
    return float(np.median(Ks))


def indices(x, fs):
    x = np.ascontiguousarray(x, dtype=np.float64)
    return dict(
        lle=lle_rosenstein(x, fs),
        k01=zero_one_K(x),
        pe=float(ant.perm_entropy(x, order=5, delay=2, normalize=True)),
        sampen=float(ant.sample_entropy(x, order=2)),
        lzc=float(ant.lziv_complexity((x > np.median(x)).astype(int), normalize=True)),
        hfd=float(ant.higuchi_fd(x, kmax=10)),
    )


def surrogate_indices(x, fs, n_surr=8, seed=0):
    rng = np.random.default_rng(seed)
    d = indices(x, fs)
    S = [indices(iaaft(x, rng), fs) for _ in range(n_surr)]
    out = {}
    for k, v in d.items():
        s = np.array([s_[k] for s_ in S])
        out[k] = v
        out[k + "_sm"] = float(s.mean()); out[k + "_ss"] = float(s.std(ddof=1))
        out[k + "_z"] = float((v - s.mean()) / (s.std(ddof=1) + 1e-12))
        # rank-based one-sided p (two-sided via min) with n_surr surrogates
        r = (np.sum(s >= v) + 1) / (n_surr + 1); r2 = (np.sum(s <= v) + 1) / (n_surr + 1)
        out[k + "_p"] = float(min(1.0, 2 * min(r, r2)))
    return out

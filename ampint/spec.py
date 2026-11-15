"""Spectral helpers: PSD, individual peak frequency, aperiodic fit, band analytic signal, surrogates."""
import numpy as np
from scipy.signal import welch, butter, sosfiltfilt, sosfreqz, hilbert

FS = 250.0


def psd(x, fs=FS, nper=None):
    nper = nper or int(4 * fs)
    f, S = welch(x, fs, nperseg=nper, noverlap=nper // 2)
    return f, S


def aperiodic_fit(f, S, frange=(2.0, 40.0), exclude=((7.0, 14.0),)):
    """Robust log-log line (knee-free) fit excluding oscillatory bands: log10 S = b - chi*log10 f."""
    m = (f >= frange[0]) & (f <= frange[1])
    for lo, hi in exclude:
        m &= ~((f >= lo) & (f <= hi))
    lf, ls = np.log10(f[m]), np.log10(S[m])
    for _ in range(3):  # iteratively down-weight peaks above the line
        A = np.vstack([np.ones_like(lf), lf]).T
        coef, *_ = np.linalg.lstsq(A, ls, rcond=None)
        res = ls - A @ coef
        keep = res < np.percentile(res, 80)
        A2 = A[keep]; coef, *_ = np.linalg.lstsq(A2, ls[keep], rcond=None)
    b, slope = coef
    return b, -slope


def peak_freq(f, S, band=(7.0, 14.0)):
    b, chi = aperiodic_fit(f, S)
    m = (f >= band[0]) & (f <= band[1])
    resid = np.log10(S[m]) - (b - chi * np.log10(f[m]))
    return float(f[m][np.argmax(resid)]), float(resid.max())


def band_sos(lo, hi, fs=FS, order=4):
    return butter(order, [lo, hi], btype="band", fs=fs, output="sos")


def analytic(x, lo, hi, fs=FS):
    return hilbert(sosfiltfilt(band_sos(lo, hi, fs), x))


def noise_fraction(f, S, lo, hi, fs=FS):
    """rho = aperiodic power / total power, both weighted by the (zero-phase) filter response."""
    b, chi = aperiodic_fit(f, S)
    A = 10 ** (b - chi * np.log10(np.maximum(f, 1e-3)))
    w, h = sosfreqz(band_sos(lo, hi, fs), worN=f, fs=fs)
    H2 = np.abs(h) ** 4  # filtfilt -> squared magnitude applied twice
    num = np.sum(np.minimum(A, S) * H2); den = np.sum(S * H2)
    return float(num / den)


def clean_runs(mask, min_len):
    """(start, stop) index pairs of contiguous True runs of length >= min_len."""
    m = np.concatenate([[False], mask, [False]]).astype(int)
    d = np.diff(m); st = np.where(d == 1)[0]; en = np.where(d == -1)[0]
    return [(a, b) for a, b in zip(st, en) if b - a >= min_len]


def run_power(x, mask, lo, hi, fs=FS, min_len_s=8.0, trim_s=1.0, surrogate_rng=None):
    """Band power |analytic|^2 pooled over clean runs (filter applied per run, edges trimmed).
    With surrogate_rng, each run is first replaced by its IAAFT surrogate. Returns list of arrays."""
    out = []
    tr = int(trim_s * fs)
    for a, b in clean_runs(mask, int(min_len_s * fs)):
        seg = x[a:b] - x[a:b].mean()
        if surrogate_rng is not None:
            seg = iaaft(seg, surrogate_rng)
        an = analytic(seg, lo, hi, fs)[tr:-tr]
        out.append(np.abs(an) ** 2)
    return out


def iaaft(x, rng, n_iter=100):
    """Iterative amplitude-adjusted Fourier-transform surrogate (Schreiber & Schmitz 1996)."""
    xs = np.sort(x); amp = np.abs(np.fft.rfft(x))
    y = rng.permutation(x)
    for _ in range(n_iter):
        Y = np.fft.rfft(y)
        y = np.fft.irfft(amp * np.exp(1j * np.angle(Y)), x.size)
        y = xs[np.argsort(np.argsort(y))]
    return y

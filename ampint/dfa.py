"""Detrended fluctuation analysis (DFA) of the amplitude envelope (Linkenkaer-Hansen et al. 2001; Hardstone et al.
2012), computed within artifact-free runs so that no window spans a run boundary."""
import numpy as np


def dfa_exponent(power_runs, fs, wmin_s=1.0, wmax_s=10.0, n_win=10, min_windows=4, min_sizes=5):
    """DFA exponent of the amplitude envelope sqrt(power) over window lengths wmin_s..wmax_s (log-spaced).
    The profile is built per run; fluctuations of all windows of a given length are pooled across runs. Window
    lengths with fewer than min_windows windows are skipped; NaN if fewer than min_sizes lengths remain."""
    amps = [np.sqrt(np.asarray(r, float)) for r in power_runs]
    wins = np.unique(np.round(np.geomspace(wmin_s, wmax_s, n_win) * fs).astype(int))
    ls, lf = [], []
    for w in wins:
        res = []
        t = np.arange(w, dtype=float)
        A = np.column_stack([t, np.ones(w)])
        P = A @ np.linalg.pinv(A)
        for a in amps:
            m = a.size // w
            if m < 1:
                continue
            y = np.cumsum(a - a.mean())[:m * w].reshape(m, w)
            r = y - y @ P.T
            res.append(np.mean(r ** 2, 1))
        if res and sum(len(r) for r in res) >= min_windows:
            ls.append(np.log10(w / fs)); lf.append(0.5 * np.log10(np.mean(np.concatenate(res))))
    if len(ls) < min_sizes:
        return np.nan
    return float(np.polyfit(ls, lf, 1)[0])

"""Rhythm bursts from band power: stretches in which the amplitude envelope exceeds a multiple of its median.
These statistics are not used for inference and serve as a posterior predictive check."""
import numpy as np

KEYS = ["dur_mean", "dur_med", "dur_p90", "frac", "rate", "gap_mean", "n"]


def burst_stats(runs, fs, f_peak, thr=1.5, min_cycles=2.0):
    """runs: list of band-power arrays (contiguous clean runs). A burst is a stretch of at least min_cycles
    cycles in which the amplitude exceeds thr x the median amplitude (pooled over runs). Bursts touching a run
    edge enter the time fraction and rate but not the duration statistics. Gaps are measured within runs."""
    amp = [np.sqrt(np.asarray(r, float)) for r in runs]
    med = np.median(np.concatenate(amp))
    min_len = int(np.ceil(min_cycles / f_peak * fs))
    durs, gaps, n_b, t_in, T = [], [], 0, 0, 0
    for a in amp:
        T += a.size
        m = np.diff(np.r_[0, (a > thr * med).astype(int), 0])
        st, en = np.where(m == 1)[0], np.where(m == -1)[0]
        keep = (en - st) >= min_len
        st, en = st[keep], en[keep]
        n_b += st.size
        t_in += int(np.sum(en - st))
        inner = (st > 0) & (en < a.size)
        durs += ((en - st)[inner] / fs).tolist()
        gaps += ((st[1:] - en[:-1]) / fs).tolist()
    durs = np.array(durs)
    return dict(dur_mean=float(durs.mean()) if durs.size else np.nan,
                dur_med=float(np.median(durs)) if durs.size else np.nan,
                dur_p90=float(np.percentile(durs, 90)) if durs.size else np.nan,
                frac=t_in / T, rate=n_b / (T / fs), gap_mean=float(np.mean(gaps)) if gaps else np.nan, n=n_b)


def all_stats(runs, fs, f_peak):
    """Burst statistics at the primary (1.5 x median amplitude) and a stricter (2 x) threshold."""
    out = {}
    for thr, pre in ((1.5, "b_"), (2.0, "b2_")):
        out.update({pre + k: v for k, v in burst_stats(runs, fs, f_peak, thr).items()})
    return out


def split_like(P, lengths):
    """Cut one long power series into consecutive pieces with the given lengths (samples)."""
    edges = np.cumsum(np.r_[0, lengths])
    return [P[a:b] for a, b in zip(edges[:-1], edges[1:]) if b <= P.size and b - a > 0]

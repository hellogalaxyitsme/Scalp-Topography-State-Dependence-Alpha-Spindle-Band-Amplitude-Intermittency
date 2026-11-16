"""Spindle-band checks on wakefulness and N2 pieces (same pieces as run_nf_anphy.py and run_nf_sleep.py):
(1) summaries in a fixed band of 11.5-14.5 Hz, independent of the selected peak, for the EEG and its linear twin
    (every second piece in ANPHY-Sleep, every fourth in Sleep-EDF);
(2) DFA exponent (1-10 s) of the amplitude envelope in the peak-centred spindle band and in the fixed band.
usage: python scripts/sleep_checks.py <anphy|sleep> <n_jobs>
       -> results/sleep_checks_<dataset>/*.csv (apply nf_apply.py with the spindle-band posterior afterwards)"""
import sys, os, numpy as np, pandas as pd, traceback
from joblib import Parallel, delayed
from scipy.signal import welch
from ampint.spec import run_power, noise_fraction, peak_freq
from ampint.summ import summaries
from ampint.dfa import dfa_exponent

FIXED = (11.5, 14.5)
SEARCH, HW = (10.5, 16.0), 1.5
STAGES = ["W", "N2"]


def piece_rows(x, fs, base, rng, with_twin):
    f, S = welch(x - x.mean(), fs, nperseg=int(4 * fs), noverlap=int(2 * fs))
    pk, pkh = peak_freq(f, S, SEARCH)
    mask = np.ones(x.size, bool)
    rp = run_power(x, mask, pk - HW, pk + HW, fs, min_len_s=30.0, trim_s=1.0)
    rho = noise_fraction(f, S, *FIXED, fs)
    rf = run_power(x, mask, *FIXED, fs, min_len_s=30.0, trim_s=1.0)
    rows = [dict(base, kind="data", peak=pk, peak_height=pkh, dfa_peak=dfa_exponent(rp, fs), dfa_fixed=dfa_exponent(rf, fs),
                 **summaries(rf, rho, fs))]
    if with_twin:
        rt = run_power(x, mask, *FIXED, fs, min_len_s=30.0, trim_s=1.0, surrogate_rng=rng)
        rows.append(dict(base, kind="iaaft0", peak=pk, peak_height=pkh, dfa_fixed=dfa_exponent(rt, fs), **summaries(rt, rho, fs)))
    return rows


def anphy_one(sid, ci, out):
    from ampint.io_anphy import load, FS
    from run_nf_anphy import chunks
    try:
        X, lab, chans, bad, mask = load(sid)
        ch = chans[ci]
        of = f"{out}/{sid}_{ch}.csv"
        if os.path.exists(of) or bad[ci]:
            return
        M = mask(ci)
        rng = np.random.default_rng(5_000 * int(sid[5:7]) + ci)
        rows = []
        for stage in STAGES:
            for k, (a, b) in enumerate(chunks(lab, M, stage)):
                base = dict(rec=sid, ch=ch, stage=stage, chunk=k)
                rows += piece_rows(np.asarray(X[ci, a:b], float), FS, base, rng, k % 2 == 0)
        pd.DataFrame(rows).to_csv(of, index=False)
    except Exception:
        with open(f"{out}/errors.log", "a") as fh:
            fh.write(f"{sid} {ci}\n" + traceback.format_exc() + "\n")


def sleep_one(psg, hyp, out):
    from ampint.io_sleepedf import load, FS, CHANS
    from run_nf_sleep import chunks
    rec = os.path.basename(psg)[:8]
    of = f"{out}/{rec}.csv"
    if os.path.exists(of):
        return
    try:
        X, lab, M = load(psg, hyp)
        rng = np.random.default_rng(9_000 + int(rec[3:6]))
        rows = []
        for ci, ch in enumerate(CHANS):
            for stage in STAGES:
                for k, (a, b) in enumerate(chunks(lab, M[ci], stage)):
                    base = dict(rec=rec, ch=ch, stage=stage, chunk=k)
                    rows += piece_rows(np.asarray(X[ci, a:b], float), FS, base, rng, k % 4 == 0)
        pd.DataFrame(rows).to_csv(of, index=False)
    except Exception:
        with open(f"{out}/errors.log", "a") as fh:
            fh.write(psg + "\n" + traceback.format_exc() + "\n")


if __name__ == "__main__":
    ds, nj = sys.argv[1], int(sys.argv[2])
    out = f"results/sleep_checks_{ds}"
    os.makedirs(out, exist_ok=True)
    if ds == "anphy":
        from ampint.io_anphy import subjects, load, CACHE
        jobs = []
        for s in [s for s in subjects() if os.path.exists(f"{CACHE}/{s}_meta.json")]:
            _, _, chans, _, _ = load(s)
            jobs += [(s, ci) for ci in range(len(chans))]
        print(len(jobs), "night-channels", flush=True)
        Parallel(n_jobs=nj, verbose=5)(delayed(anphy_one)(s, ci, out) for s, ci in jobs)
    else:
        from ampint.io_sleepedf import pairs
        pp = pairs()
        print(len(pp), "recordings", flush=True)
        Parallel(n_jobs=nj, verbose=5)(delayed(sleep_one)(a, b, out) for a, b in pp)

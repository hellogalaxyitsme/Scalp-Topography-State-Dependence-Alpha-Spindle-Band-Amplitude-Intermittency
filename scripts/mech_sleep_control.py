"""Second, independent linear twins for the sleep recordings, for the empirical negative control of the mechanism
classifier (twin minus twin) in the sleep data. For every piece that carries a first twin (iaaft0) in
run_nf_anphy.py / run_nf_sleep.py, a second IAAFT twin is generated with an independent seed and summarised with the
band, peak and noise fraction of the data, exactly as the first. ANPHY-Sleep: 20 channels spread over the scalp.
usage: python scripts/mech_sleep_control.py <anphy|sleep> <n_jobs> [rev]   (rev: process the recordings in reverse order,
       so that a second job can run alongside the first; existing outputs are skipped)
Output: results/twin2_anphy/<night>_<ch>.csv, results/twin2_sleep/<rec>.csv (kind = iaaft1; resumable)."""
import sys, os, numpy as np, pandas as pd, traceback
from joblib import Parallel, delayed
from scipy.signal import welch
from ampint.spec import run_power, noise_fraction, peak_freq
from ampint.summ import summaries

CONTROL_CH = ["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T3", "C3", "CZ", "C4", "T4",
              "T5", "P3", "PZ", "P4", "T6", "O1", "OZ", "O2"]  # ANPHY uses the 10-20 temporal labels T3-T6


def twins(x, fs, bands, base, rng):
    f, S = welch(x - x.mean(), fs, nperseg=int(4 * fs), noverlap=int(2 * fs))
    rows = []
    for bname, (search, hw) in bands.items():
        pk, pkh = peak_freq(f, S, search)
        lo, hi = pk - hw, pk + hw
        rho = noise_fraction(f, S, lo, hi, fs)
        sr = run_power(x, np.ones(x.size, bool), lo, hi, fs, min_len_s=30.0, surrogate_rng=rng)
        rows.append(dict(base, band=bname, kind="iaaft1", peak=pk, peak_height=pkh, **summaries(sr, rho, fs)))
    return rows


def anphy_one(sid, ci):
    from ampint.io_anphy import load, FS
    from run_nf_anphy import chunks, BANDS
    try:
        X, lab, chans, bad, mask = load(sid)
        ch = chans[ci]
        of = f"results/twin2_anphy/{sid}_{ch}.csv"
        if os.path.exists(of) or bad[ci]:
            return
        rng = np.random.default_rng(5_000_000 + 1000 * int(sid[5:7]) + ci)
        rows = []
        for stage in ["W", "N1", "N2", "N3", "REM"]:
            for k, (a, b) in enumerate(chunks(lab, mask(ci), stage)):
                if k % 2 == 0:
                    rows += twins(np.asarray(X[ci, a:b], float), FS, BANDS, dict(rec=sid, ch=ch, stage=stage, chunk=k), rng)
        pd.DataFrame(rows).to_csv(of, index=False)
    except Exception:
        with open("results/twin2_anphy/errors.log", "a") as fh:
            fh.write(f"{sid} {ci}\n" + traceback.format_exc() + "\n")


def sleep_one(psg, hyp):
    from ampint.io_sleepedf import load, FS, CHANS
    from run_nf_sleep import chunks, BANDS
    rec = os.path.basename(psg)[:8]
    of = f"results/twin2_sleep/{rec}.csv"
    if os.path.exists(of):
        return
    try:
        X, lab, M = load(psg, hyp)
        rng = np.random.default_rng(7_000_000 + int(rec[3:6]))
        rows = []
        for ci, ch in enumerate(CHANS):
            for stage in ["W", "N1", "N2", "N3", "REM"]:
                for k, (a, b) in enumerate(chunks(lab, M[ci], stage)):
                    if k % 4 == 0:
                        rows += twins(X[ci, a:b], FS, BANDS, dict(rec=rec, ch=ch, stage=stage, chunk=k), rng)
        pd.DataFrame(rows).to_csv(of, index=False)
    except Exception:
        with open("results/twin2_sleep/errors.log", "a") as fh:
            fh.write(psg + "\n" + traceback.format_exc() + "\n")


if __name__ == "__main__":
    mode, nj = sys.argv[1], int(sys.argv[2])
    os.makedirs(f"results/twin2_{mode}", exist_ok=True)
    if mode == "anphy":
        from ampint.io_anphy import subjects, load, CACHE
        jobs = []
        for s in [s for s in subjects() if os.path.exists(f"{CACHE}/{s}_meta.json")]:
            chans = load(s)[2]
            jobs += [(s, chans.index(c)) for c in CONTROL_CH if c in chans]
        print(len(jobs), "night-channels", flush=True)
        Parallel(n_jobs=nj, verbose=5)(delayed(anphy_one)(s, ci) for s, ci in jobs)
    else:
        from ampint.io_sleepedf import pairs
        pp = pairs()[::-1] if len(sys.argv) > 3 and sys.argv[3] == "rev" else pairs()
        print(len(pp), "recordings", flush=True)
        Parallel(n_jobs=nj, verbose=5)(delayed(sleep_one)(a, b) for a, b in pp)

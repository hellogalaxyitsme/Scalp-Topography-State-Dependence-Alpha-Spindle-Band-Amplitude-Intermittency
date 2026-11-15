"""External MDD replication (Mumtaz et al. 2017 dataset): normal-form summaries in 100-s pieces + chaos indices."""
import sys, os, re, glob, numpy as np, pandas as pd, traceback, mne
from joblib import Parallel, delayed
from scipy.signal import decimate
from ampint.spec import clean_runs, run_power, noise_fraction, peak_freq
from ampint.pipeline import clean_psd
from ampint.summ import summaries
from ampint.chaos import surrogate_indices
from run_chaos_tdbrain import spectral_covariates

from ampint.paths import MUMTAZ_ROOT as ROOT
OUT = "results/mumtaz"
FS = 250.0
CHAOS_CH = ["Fz", "Cz", "Pz", "O1"]
PIECE = 100.0


def load(path, thr_uv=150.0, pad_s=0.5, edge_s=3.0):
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    eeg = [c for c in raw.ch_names if c.endswith("-LE")]
    raw.pick(eeg); raw.rename_channels({c: c.replace("EEG ", "").replace("-LE", "") for c in eeg})
    raw.filter(0.5, 45.0, verbose="ERROR"); raw.resample(FS, verbose="ERROR")
    X = raw.get_data() * 1e6; n = X.shape[1]
    M = np.abs(X) <= thr_uv; k = int(pad_s * FS)
    for i in range(X.shape[0]):
        if not M[i].all():
            M[i] = np.convolve((~M[i]).astype(float), np.ones(2 * k + 1), "same") == 0
    e = int(edge_s * FS); M[:, :e] = False; M[:, n - e:] = False
    ls = np.log(X[:, e:n - e].std(1) + 1e-9); med = np.median(ls); mad = np.median(np.abs(ls - med)) * 1.4826 + 1e-9
    badch = (np.abs(ls - med) / mad > 3.5) | (M[:, e:n - e].mean(1) < 0.7)
    return X, M, badch, raw.ch_names


def pieces(mask, piece_s=PIECE, min_s=60.0):
    """Group clean runs (>=8 s) into consecutive pieces of ~piece_s seconds of clean data."""
    runs = clean_runs(mask, int(8 * FS)); out, cur, tot = [], [], 0
    for a, b in runs:
        while b - a > 0:
            take = min(b - a, int(piece_s * FS) - tot)
            cur.append((a, a + take)); tot += take; a += take
            if tot >= piece_s * FS:
                out.append(cur); cur, tot = [], 0
    if tot >= min_s * FS:
        out.append(cur)
    return out


def one(path):
    fn = os.path.basename(path)
    grp = "MDD" if fn.startswith("MDD") else "HEALTHY"
    sid = re.search(r"S(\d+)", fn).group(1); task = "restEC" if fn.strip().endswith("EC.edf") else "restEO"
    of = f"{OUT}/{grp}_{sid}_{task}.csv"
    if os.path.exists(of):
        return
    try:
        X, M, badch, chs = load(path)
        rows = []
        for ci, ch in enumerate(chs):
            if badch[ci]:
                continue
            f, S = clean_psd(X[ci], M[ci], FS)
            if f is None:
                continue
            pk, pkh = peak_freq(f, S, (7.0, 14.0)); lo, hi = pk - 2.0, pk + 2.0
            rho = noise_fraction(f, S, lo, hi, FS)
            cov = spectral_covariates(f, S)
            for pi, pc in enumerate(pieces(M[ci])):
                m = np.zeros(X.shape[1], bool)
                for a, b in pc:
                    m[a:b] = True
                runs = run_power(X[ci], m, lo, hi, FS)
                if not runs:
                    continue
                s = summaries(runs, rho, FS)
                rows.append(dict(group=grp, sub=f"{grp}{sid}", task=task, ch=ch, piece=pi, kind="nf", peak=pk, peak_height=pkh, **cov, **s))
            if ch in CHAOS_CH:
                segs = []
                for a, b in clean_runs(M[ci], int(20 * FS)):
                    k = a
                    while k + int(20 * FS) <= b and len(segs) < 2:
                        segs.append(X[ci, k:k + int(20 * FS)]); k += int(20 * FS)
                for si, seg in enumerate(segs):
                    y = decimate(seg - seg.mean(), 2, zero_phase=True)
                    rows.append(dict(group=grp, sub=f"{grp}{sid}", task=task, ch=ch, piece=si, kind="chaos", **cov,
                                     **surrogate_indices(y, FS / 2, n_surr=10, seed=si)))
        pd.DataFrame(rows).to_csv(of, index=False)
    except Exception:
        with open(f"{OUT}/errors.log", "a") as fh:
            fh.write(path + "\n" + traceback.format_exc() + "\n")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    files = [f for f in sorted(glob.glob(f"{ROOT}/*.edf")) if not f.endswith("TASK.edf")]
    print(len(files), "files", flush=True)
    Parallel(n_jobs=int(sys.argv[1]), verbose=5)(delayed(one)(p) for p in files)

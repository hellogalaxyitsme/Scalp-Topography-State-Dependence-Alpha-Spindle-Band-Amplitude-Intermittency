"""Surrogate-controlled chaos/complexity indices + spectral covariates, TDBRAIN V3.1 rest EC/EO."""
import sys, os, glob, numpy as np, pandas as pd, traceback
from joblib import Parallel, delayed
from scipy.signal import decimate
from ampint.io_tdbrain import load_rest, ROOT, FS
from ampint.spec import clean_runs, aperiodic_fit, peak_freq
from ampint.pipeline import clean_psd
from ampint.chaos import surrogate_indices

CH = ["Fz", "Cz", "Pz", "Oz"]
OUT = "results/chaos_tdbrain"
SEG_S = 20.0


def spectral_covariates(f, S):
    b, chi = aperiodic_fit(f, S)
    iaf, pk = peak_freq(f, S)
    tot = S[(f >= 1) & (f <= 40)].sum()
    rel = {k: float(S[(f >= lo) & (f < hi)].sum() / tot) for k, (lo, hi) in
           dict(delta=(1, 4), theta=(4, 8), alpha=(8, 13), beta=(13, 30), gamma=(30, 40)).items()}
    return dict(ap_offset=b, ap_exp=chi, iaf=iaf, alpha_peak=pk, **{"rel_" + k: v for k, v in rel.items()})


def one(path):
    sub, ses, task = os.path.basename(path).split("_")[:3]
    task = task.replace("task-", "")
    of = f"{OUT}/{sub}_{ses}_{task}.csv"
    if os.path.exists(of):
        return
    try:
        X, M, badch, picks = load_rest(path)
        rows = []
        for ch in CH:
            ci = picks.index(ch)
            if badch[ci]:
                continue
            f, S = clean_psd(X[ci], M[ci], FS)
            if f is None:
                continue
            cov = spectral_covariates(f, S)
            segs = []
            for a, b in clean_runs(M[ci], int(SEG_S * FS)):
                k = a
                while k + int(SEG_S * FS) <= b and len(segs) < 2:
                    segs.append(X[ci, k:k + int(SEG_S * FS)]); k += int(SEG_S * FS)
            for si, seg in enumerate(segs):
                y = decimate(seg - seg.mean(), 2, zero_phase=True)  # 125 Hz
                r = surrogate_indices(y, FS / 2, n_surr=10, seed=si)
                rows.append(dict(sub=sub, ses=ses, task=task, ch=ch, seg=si, **cov, **r))
        pd.DataFrame(rows).to_csv(of, index=False)
    except Exception:
        with open(f"{OUT}/errors.log", "a") as fh:
            fh.write(path + "\n" + traceback.format_exc() + "\n")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(f"{ROOT}/sub-*/ses-*/eeg/*task-rest*_eeg.bdf"))
    print(len(files), "files", flush=True)
    Parallel(n_jobs=int(sys.argv[1]) if len(sys.argv) > 1 else 10, verbose=5)(delayed(one)(p) for p in files)

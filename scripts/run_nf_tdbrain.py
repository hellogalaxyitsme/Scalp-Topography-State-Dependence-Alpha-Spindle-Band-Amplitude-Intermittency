"""Normal-form summary statistics for every TDBRAIN V3.1 rest recording and channel (alpha band),
computed exactly as in the simulation bank, plus IAAFT-surrogate summaries on posterior channels."""
import sys, os, glob, numpy as np, pandas as pd, traceback
from joblib import Parallel, delayed
from ampint.io_tdbrain import load_rest, ROOT, FS
from ampint.spec import run_power, noise_fraction, peak_freq
from ampint.pipeline import clean_psd
from ampint.summ import summaries

OUT = "results/nf_tdbrain"
SURR_CH = ["O1", "Oz", "O2", "Pz"]
SEARCH, HW = (7.0, 14.0), 2.0


def one(path):
    sub, ses, task = os.path.basename(path).split("_")[:3]
    task = task.replace("task-", "")
    of = f"{OUT}/{sub}_{ses}_{task}.csv"
    if os.path.exists(of):
        return
    try:
        X, M, badch, picks = load_rest(path)
        rows = []
        rng = np.random.default_rng(abs(hash(of)) % 2**32)
        for ci, ch in enumerate(picks):
            if badch[ci]:
                continue
            f, S = clean_psd(X[ci], M[ci], FS)
            if f is None:
                continue
            pk, pkh = peak_freq(f, S, SEARCH)
            lo, hi = pk - HW, pk + HW
            rho = noise_fraction(f, S, lo, hi, FS)
            runs = run_power(X[ci], M[ci], lo, hi, FS)
            if sum(r.size for r in runs) < 40 * FS:
                continue
            s = summaries(runs, rho, FS)
            rows.append(dict(sub=sub, ses=ses, task=task, ch=ch, kind="data", peak=pk, peak_height=pkh, **s))
            if ch in SURR_CH:
                for k in range(2):
                    sr = run_power(X[ci], M[ci], lo, hi, FS, surrogate_rng=rng)
                    ss = summaries(sr, rho, FS)
                    rows.append(dict(sub=sub, ses=ses, task=task, ch=ch, kind=f"iaaft{k}", peak=pk, peak_height=pkh, **ss))
        pd.DataFrame(rows).to_csv(of, index=False)
    except Exception:
        with open(f"{OUT}/errors.log", "a") as fh:
            fh.write(path + "\n" + traceback.format_exc() + "\n")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(f"{ROOT}/sub-*/ses-*/eeg/*task-rest*_eeg.bdf"))
    if len(sys.argv) > 2:
        files = files[:int(sys.argv[2])]
    print(len(files), "files", flush=True)
    Parallel(n_jobs=int(sys.argv[1]), verbose=5)(delayed(one)(p) for p in files)

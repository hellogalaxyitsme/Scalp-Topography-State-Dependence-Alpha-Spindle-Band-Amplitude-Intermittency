"""Third cohort (OpenNeuro ds003478, Cavanagh/Allen lab): students with high vs low BDI, same lab/procedure.
First rest run (6 x 1-min blocks, eyes closed/open). Linked-mastoid re-reference to match TDBRAIN.
Normal-form summaries in ~100-s pieces + chaos indices (Fz, Cz, Pz, Oz)."""
import sys, os, glob, numpy as np, pandas as pd, traceback, mne
from joblib import Parallel, delayed
from scipy.signal import decimate
from ampint.spec import clean_runs, run_power, noise_fraction, peak_freq
from ampint.pipeline import clean_psd
from ampint.summ import summaries
from ampint.chaos import surrogate_indices
from run_chaos_tdbrain import spectral_covariates
from run_mumtaz import pieces

from ampint.paths import DS003478_ROOT as ROOT
OUT = "results/ds003478"
FS = 250.0
NF_CH = ["O1", "OZ", "O2", "PZ", "P3", "P4", "POZ", "PO3", "PO4", "FZ", "CZ"]
CHAOS_CH = ["FZ", "CZ", "PZ", "OZ"]
EXCLUDE_ORIG = {544}  # README: unstable BDI between mass and lab assessment


def load(path, ev_path, thr_uv=150.0, pad_s=0.5):
    raw = mne.io.read_raw_eeglab(path, preload=True, verbose="ERROR")
    if "M1" in raw.ch_names and "M2" in raw.ch_names:
        raw.set_eeg_reference(["M1", "M2"], verbose="ERROR")
    raw.pick([c for c in NF_CH if c in raw.ch_names])
    raw.filter(0.5, 45.0, verbose="ERROR"); fs0 = raw.info["sfreq"]; raw.resample(FS, verbose="ERROR")
    X = raw.get_data() * 1e6; n = X.shape[1]
    ev = pd.read_csv(ev_path, sep="\t")
    ev["value"] = pd.to_numeric(ev["value"], errors="coerce")
    starts = ev[ev.value.isin([11, 12, 13, 14])].copy()
    # a block starts at the first marker of a new code group; keep onsets where code changes
    starts = starts[starts.value.ne(starts.value.shift())]
    lab = np.full(n, "?", dtype=object)
    on = (starts.onset.values * FS).astype(int); code = starts.value.values
    for i in range(len(on)):
        a = on[i]; b = on[i + 1] if i + 1 < len(on) else min(n, a + int(62 * FS))
        b = min(b, a + int(62 * FS))
        lab[a:b] = "restEC" if code[i] in (11, 13) else "restEO"
    M = np.abs(X) <= thr_uv; k = int(pad_s * FS)
    for i in range(X.shape[0]):
        if not M[i].all():
            M[i] = np.convolve((~M[i]).astype(float), np.ones(2 * k + 1), "same") == 0
    return X, M, lab, raw.ch_names


def one(sub, orig, bdi):
    of = f"{OUT}/{sub}.csv"
    if os.path.exists(of):
        return
    try:
        f = f"{ROOT}/{sub}/eeg/{sub}_task-Rest_run-01_eeg.set"
        X, M, lab, chs = load(f, f.replace("_eeg.set", "_events.tsv"))
        rows = []
        for task in ["restEC", "restEO"]:
            for ci, ch in enumerate(chs):
                m = M[ci] & (lab == task)
                if m.sum() < 60 * FS:
                    continue
                fr, S = clean_psd(X[ci], m, FS)
                if fr is None:
                    continue
                pk, pkh = peak_freq(fr, S, (7.0, 14.0)); lo, hi = pk - 2.0, pk + 2.0
                rho = noise_fraction(fr, S, lo, hi, FS); cov = spectral_covariates(fr, S)
                for pi, pc in enumerate(pieces(m)):
                    mm = np.zeros(X.shape[1], bool)
                    for a, b in pc:
                        mm[a:b] = True
                    runs = run_power(X[ci], mm, lo, hi, FS)
                    if runs:
                        rows.append(dict(sub=sub, orig=orig, BDI=bdi, task=task, ch=ch, piece=pi, kind="nf", peak=pk,
                                         peak_height=pkh, **cov, **summaries(runs, rho, FS)))
                if ch in CHAOS_CH:
                    segs = []
                    for a, b in clean_runs(m, int(20 * FS)):
                        kk = a
                        while kk + int(20 * FS) <= b and len(segs) < 2:
                            segs.append(X[ci, kk:kk + int(20 * FS)]); kk += int(20 * FS)
                    for si, seg in enumerate(segs):
                        y = decimate(seg - seg.mean(), 2, zero_phase=True)
                        rows.append(dict(sub=sub, orig=orig, BDI=bdi, task=task, ch=ch, piece=si, kind="chaos", **cov,
                                         **surrogate_indices(y, FS / 2, n_surr=10, seed=si)))
        pd.DataFrame(rows).to_csv(of, index=False)
    except Exception:
        with open(f"{OUT}/errors.log", "a") as fh:
            fh.write(sub + "\n" + traceback.format_exc() + "\n")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    p = pd.read_csv(f"{ROOT}/participants.tsv", sep="\t")
    p = p[~p.Original_ID.isin(EXCLUDE_ORIG)]
    jobs = [(r.participant_id, r.Original_ID, r.BDI) for r in p.itertuples()]
    if len(sys.argv) > 2:
        jobs = jobs[:int(sys.argv[2])]
    print(len(jobs), "participants", flush=True)
    Parallel(n_jobs=int(sys.argv[1]), verbose=5)(delayed(one)(*j) for j in jobs)

"""TDBRAIN checks on O1, Oz, O2 and Pz:
(1) linear twins with re-estimated parameters: for each clean run one IAAFT twin is generated, and its summaries are
    computed twice, with the peak frequency and background fraction of the EEG channel (as in run_nf_tdbrain.py) and
    with the peak frequency and background fraction re-estimated on the twin itself (healthy adults, first session);
(2) DFA exponent (1-10 s) of the alpha amplitude envelope of the EEG and of the twin, for the healthy reference
    sample and for all participants with two sessions.
usage: python scripts/tdbrain_checks.py <n_jobs>  -> results/tdbrain_checks/*.csv (apply nf_apply.py afterwards)"""
import sys, os, glob, numpy as np, pandas as pd, traceback
from joblib import Parallel, delayed
from ampint.io_tdbrain import load_rest, ROOT, FS
from ampint.spec import run_power, noise_fraction, peak_freq, clean_runs, iaaft
from ampint.pipeline import clean_psd
from ampint.summ import summaries
from ampint.dfa import dfa_exponent

OUT = "results/tdbrain_checks"
CH = ["O1", "Oz", "O2", "Pz"]
SEARCH, HW = (7.0, 14.0), 2.0


def twin_signal(x, mask, rng):
    y = np.array(x, float, copy=True)
    for a, b in clean_runs(mask, int(8.0 * FS)):
        seg = y[a:b] - y[a:b].mean()
        y[a:b] = iaaft(seg, rng)
    return y


def one(path, with_twin, seed):
    sub, ses, task = os.path.basename(path).split("_")[:3]
    task = task.replace("task-", "")
    of = f"{OUT}/{sub}_{ses}_{task}.csv"
    if os.path.exists(of):
        return
    try:
        X, M, badch, picks = load_rest(path)
        rng = np.random.default_rng(seed)
        rows = []
        for ch in CH:
            ci = picks.index(ch)
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
            base = dict(sub=sub, ses=ses, task=task, ch=ch)
            rows.append(dict(base, kind="data", peak=pk, peak_height=pkh, rho_used=rho, dfa=dfa_exponent(runs, FS)))
            if with_twin:
                y = twin_signal(X[ci], M[ci], rng)
                rt = run_power(y, M[ci], lo, hi, FS)
                rows.append(dict(base, kind="twin_matched", peak=pk, peak_height=pkh, rho_used=rho, dfa=dfa_exponent(rt, FS),
                                 **summaries(rt, rho, FS)))
                f2, S2 = clean_psd(y, M[ci], FS)
                pk2, pkh2 = peak_freq(f2, S2, SEARCH)
                rho2 = noise_fraction(f2, S2, pk2 - HW, pk2 + HW, FS)
                ro = run_power(y, M[ci], pk2 - HW, pk2 + HW, FS)
                rows.append(dict(base, kind="twin_own", peak=pk2, peak_height=pkh2, rho_used=rho2, dfa=dfa_exponent(ro, FS),
                                 **summaries(ro, rho2, FS)))
        pd.DataFrame(rows).to_csv(of, index=False)
    except Exception:
        with open(f"{OUT}/errors.log", "a") as fh:
            fh.write(path + "\n" + traceback.format_exc() + "\n")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    R = pd.read_csv("results/nf_tdbrain_reclevel.csv")
    healthy = set(R.loc[R.group == "HEALTHY", "sub"])
    P = pd.read_csv("results/post_tdbrain.csv", usecols=["sub", "ses", "kind"])
    P = P[P.kind == "data"]
    retest = set(P[P.ses == "ses-1"]["sub"]) & set(P[P.ses == "ses-2"]["sub"])
    jobs = []
    for k, path in enumerate(sorted(glob.glob(f"{ROOT}/sub-*/ses-*/eeg/*task-rest*_eeg.bdf"))):
        sub, ses = os.path.basename(path).split("_")[:2]
        h1 = sub in healthy and ses == "ses-1"
        rt = sub in retest and ses in ("ses-1", "ses-2")
        if h1 or rt:
            jobs.append((path, h1, 7_000 + k))
    print(len(jobs), "recordings", flush=True)
    Parallel(n_jobs=int(sys.argv[1]), verbose=5)(delayed(one)(*j) for j in jobs)

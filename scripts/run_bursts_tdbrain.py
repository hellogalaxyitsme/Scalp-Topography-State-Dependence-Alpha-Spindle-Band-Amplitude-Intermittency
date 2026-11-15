"""Alpha burst statistics (TDBRAIN V3.1, eyes closed and open) on the channels with linear twins, for the EEG
and one fresh IAAFT twin. Band and preprocessing exactly as in run_nf_tdbrain.py.
usage: python scripts/run_bursts_tdbrain.py <n_jobs>           healthy adults, first session  -> results/bursts_tdbrain
       python run_bursts_tdbrain.py <n_jobs> retest    all participants with two sessions, both sessions
                                                       -> results/bursts_retest"""
import sys, os, numpy as np, pandas as pd, traceback
from joblib import Parallel, delayed
from ampint.io_tdbrain import load_rest, load_meta, rest_file, FS
from ampint.spec import run_power, noise_fraction, peak_freq
from ampint.pipeline import clean_psd
from ampint.bursts import all_stats

CH = ["O1", "Oz", "O2", "Pz"]
SEARCH, HW = (7.0, 14.0), 2.0


def one(sub, task, ses, out):
    of = f"{out}/{sub}_{task}.csv" if out.endswith("tdbrain") else f"{out}/{sub}_ses-{ses}_{task}.csv"
    path = rest_file(sub, task, ses)
    if os.path.exists(of) or path is None:
        return
    try:
        X, M, badch, picks = load_rest(path)
        rng = np.random.default_rng(int(sub.replace("sub-", "")) % 2**31 + (0 if task == "restEC" else 7) + 100 * (ses - 1))
        rows = []
        for ci, ch in enumerate(picks):
            if ch not in CH or badch[ci]:
                continue
            f, S = clean_psd(X[ci], M[ci], FS)
            if f is None:
                continue
            pk, _ = peak_freq(f, S, SEARCH)
            lo, hi = pk - HW, pk + HW
            runs = run_power(X[ci], M[ci], lo, hi, FS)
            if sum(r.size for r in runs) < 40 * FS:
                continue
            base = dict(sub=sub, ses=f"ses-{ses}", task=task, ch=ch, peak=pk, rho=noise_fraction(f, S, lo, hi, FS),
                        run_lengths="|".join(str(r.size) for r in runs))
            rows.append(dict(base, kind="data", **all_stats(runs, FS, pk)))
            sr = run_power(X[ci], M[ci], lo, hi, FS, surrogate_rng=rng)
            rows.append(dict(base, kind="twin", **all_stats(sr, FS, pk)))
        pd.DataFrame(rows).to_csv(of, index=False)
    except Exception:
        with open(f"{out}/errors.log", "a") as fh:
            fh.write(f"{sub} {task} {ses}\n" + traceback.format_exc() + "\n")


if __name__ == "__main__":
    retest = len(sys.argv) > 2 and sys.argv[2] == "retest"
    out = "results/bursts_retest" if retest else "results/bursts_tdbrain"
    os.makedirs(out, exist_ok=True)
    if retest:
        p = pd.read_csv("results/post_tdbrain.csv", usecols=["sub", "ses", "task", "kind"])
        p = p[p.kind == "data"].drop_duplicates(["sub", "ses", "task"])
        both = p.groupby(["sub", "task"]).ses.apply(lambda s: {"ses-1", "ses-2"} <= set(s))
        jobs = [(s, t, k) for (s, t), ok in both.items() if ok for k in (1, 2)]
    else:
        meta = load_meta()
        m1 = meta[(meta.sessID == 1) & (meta.indication == "HEALTHY") & (meta.age >= 18) & meta.gender.notna()].drop_duplicates("sub")
        jobs = [(s, t, 1) for s in m1["sub"] for t in ("restEC", "restEO")]
    print(len(jobs), "recordings", flush=True)
    Parallel(n_jobs=int(sys.argv[1]), verbose=5)(delayed(one)(s, t, k, out) for s, t, k in jobs)

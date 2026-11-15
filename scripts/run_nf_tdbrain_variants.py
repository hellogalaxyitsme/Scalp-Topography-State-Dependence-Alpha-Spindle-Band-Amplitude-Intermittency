"""Sensitivity variants of the TDBRAIN normal-form summaries for healthy adults (first session, eyes closed and
open): 'eog' (EOG regressed out) and 'csd' (surface Laplacian). Otherwise identical to run_nf_tdbrain.py.
usage: python scripts/run_nf_tdbrain_variants.py <variant> <n_jobs>"""
import sys, os, numpy as np, pandas as pd, traceback
from joblib import Parallel, delayed
from ampint.io_tdbrain import load_rest, load_meta, rest_file, FS
from ampint.spec import run_power, noise_fraction, peak_freq
from ampint.pipeline import clean_psd
from ampint.summ import summaries

SURR_CH = ["O1", "Oz", "O2", "Pz"]
SEARCH, HW = (7.0, 14.0), 2.0


def one(variant, sub, task):
    out = f"results/nf_tdbrain_{variant}"
    of = f"{out}/{sub}_ses-1_{task}.csv"
    path = rest_file(sub, task, 1)
    if os.path.exists(of) or path is None:
        return
    try:
        X, M, badch, picks = load_rest(path, variant=variant)
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
            rows.append(dict(sub=sub, ses="ses-1", task=task, ch=ch, kind="data", peak=pk, peak_height=pkh, **summaries(runs, rho, FS)))
            if ch in SURR_CH:
                for k in range(2):
                    sr = run_power(X[ci], M[ci], lo, hi, FS, surrogate_rng=rng)
                    rows.append(dict(sub=sub, ses="ses-1", task=task, ch=ch, kind=f"iaaft{k}", peak=pk, peak_height=pkh, **summaries(sr, rho, FS)))
        pd.DataFrame(rows).to_csv(of, index=False)
    except Exception:
        with open(f"{out}/errors.log", "a") as fh:
            fh.write(f"{sub} {task}\n" + traceback.format_exc() + "\n")


if __name__ == "__main__":
    variant, nj = sys.argv[1], int(sys.argv[2])
    assert variant in ("eog", "csd")
    os.makedirs(f"results/nf_tdbrain_{variant}", exist_ok=True)
    meta = load_meta()
    m1 = meta[(meta.sessID == 1) & (meta.indication == "HEALTHY") & (meta.age >= 18) & meta.gender.notna()].drop_duplicates("sub")
    jobs = [(s, t) for s in m1["sub"] for t in ("restEC", "restEO")]
    print(len(jobs), "recordings", flush=True)
    Parallel(n_jobs=nj, verbose=5)(delayed(one)(variant, s, t) for s, t in jobs)

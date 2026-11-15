"""ANPHY-Sleep (high-density replication): normal-form summaries of the sigma (spindle) and alpha rhythms
for every scalp channel and sleep stage, on contiguous same-stage chunks of 60-112 s as for Sleep-EDF.
usage: python scripts/run_nf_anphy.py prep <n_jobs>   (convert each night once; see ampint/io_anphy.py)
       python run_nf_anphy.py summ <n_jobs>   (summaries per night and channel)"""
import sys, os, numpy as np, pandas as pd, traceback
from joblib import Parallel, delayed
from scipy.signal import welch
from ampint.io_anphy import subjects, prepare, load, head_positions, FS, CACHE
from ampint.csd import operator, support
from ampint.spec import clean_runs, run_power, noise_fraction, peak_freq, aperiodic_fit
from ampint.summ import summaries

OUT = "results/nf_anphy"
BANDS = {"alpha": ((7.0, 14.0), 2.0), "sigma250": ((10.5, 16.0), 1.5)}
CHUNK = (60.0, 112.0)
MAX_CHUNKS = 6  # per stage, spread evenly over the night
BIPOLAR = [("FPZ", "CZ"), ("PZ", "OZ")]  # the two Sleep-EDF derivations, for direct comparison


def chunks(lab, M, stage):
    ok = (lab == stage) & M
    out = []
    for a, b in clean_runs(ok, int(CHUNK[0] * FS)):
        nchunk = max(1, int(np.ceil((b - a) / (CHUNK[1] * FS))))
        edges = np.linspace(a, b, nchunk + 1).astype(int)
        out += [(edges[i], edges[i + 1]) for i in range(nchunk) if edges[i + 1] - edges[i] >= CHUNK[0] * FS]
    if len(out) > MAX_CHUNKS:
        out = [out[i] for i in np.linspace(0, len(out) - 1, MAX_CHUNKS).round().astype(int)]
    return out


def one(sid, ci, csd=False):
    """ci: channel index, or a pair of channel names for a bipolar derivation.
    csd: surface Laplacian of the good scalp channels (artefact mask: all contributing channels clean)."""
    out = OUT + ("_csd" if csd else "")
    try:
        X, lab, chans, bad, mask = load(sid)
        if csd:
            hp, _ = head_positions()
            g = [i for i, c in enumerate(chans) if not bad[i] and c in hp]
            if ci not in g:
                return
            W = operator([chans[i] for i in g], hp); r = g.index(ci)
            ch, skip, seed = chans[ci], False, ci
            M = np.logical_and.reduce([mask(g[j]) for j in np.where(support(W)[r])[0]])
            sig = lambda a, b: W[r] @ np.asarray(X[g, a:b], float)
        elif isinstance(ci, tuple):
            ia, ib = chans.index(ci[0]), chans.index(ci[1])
            ch, skip, seed = f"{ci[0]}-{ci[1]}", bad[ia] or bad[ib], 900 + ia
            M = mask(ia) & mask(ib)
            sig = lambda a, b: np.asarray(X[ia, a:b], float) - np.asarray(X[ib, a:b], float)
        else:
            ch, skip, seed = chans[ci], bad[ci], ci
            M = mask(ci)
            sig = lambda a, b: np.asarray(X[ci, a:b], float)
        of = f"{out}/{sid}_{ch}.csv"
        if os.path.exists(of) or skip:
            return
        rng = np.random.default_rng(1000 * int(sid[5:7]) + seed)
        rows = []
        for stage in ["W", "N1", "N2", "N3", "REM"]:
            for k, (a, b) in enumerate(chunks(lab, M, stage)):
                x = sig(a, b)
                f, S = welch(x - x.mean(), FS, nperseg=int(4 * FS), noverlap=int(2 * FS))
                base = dict(rec=sid, ch=ch, stage=stage, chunk=k, t_start_h=a / FS / 3600, dur=(b - a) / FS)
                for bname, (search, hw) in BANDS.items():
                    pk, pkh = peak_freq(f, S, search)
                    lo, hi = pk - hw, pk + hw
                    rho = noise_fraction(f, S, lo, hi, FS)
                    runs = run_power(x, np.ones(x.size, bool), lo, hi, FS, min_len_s=30.0, trim_s=1.0)
                    rows.append(dict(base, band=bname, kind="data", peak=pk, peak_height=pkh, **summaries(runs, rho, FS)))
                    if k % 2 == 0:  # linear twins on every second chunk
                        sr = run_power(x, np.ones(x.size, bool), lo, hi, FS, min_len_s=30.0, surrogate_rng=rng)
                        rows.append(dict(base, band=bname, kind="iaaft0", peak=pk, peak_height=pkh, **summaries(sr, rho, FS)))
        pd.DataFrame(rows).to_csv(of, index=False)
    except Exception:
        with open(f"{out}/errors.log", "a") as fh:
            fh.write(f"{sid} {ci}\n" + traceback.format_exc() + "\n")


if __name__ == "__main__":
    mode, nj = sys.argv[1], int(sys.argv[2])
    csd = len(sys.argv) > 3 and sys.argv[3] == "csd"
    sids = subjects()
    print(len(sids), "nights", flush=True)
    if mode == "prep":
        Parallel(n_jobs=nj, verbose=5)(delayed(prepare)(s) for s in sids)
    else:
        os.makedirs(OUT + ("_csd" if csd else ""), exist_ok=True)
        jobs = []
        for s in [s for s in sids if os.path.exists(f"{CACHE}/{s}_meta.json")]:  # converted nights; rerun to add the rest
            _, _, chans, _, _ = load(s)
            jobs += ([] if csd else [(s, bp) for bp in BIPOLAR]) + [(s, ci) for ci in range(len(chans))]
        print(len(jobs), "night-channels", flush=True)
        Parallel(n_jobs=nj, verbose=5)(delayed(one)(s, ci, csd) for s, ci in jobs)

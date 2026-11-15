"""Sleep-EDF: normal-form summaries (alpha100 & sigma configs) and surrogate-controlled chaos indices
per stage, on contiguous same-stage chunks of 60-300 s (matching the simulation-bank length prior)."""
import sys, os, numpy as np, pandas as pd, traceback
from joblib import Parallel, delayed
from scipy.signal import welch, decimate
from ampint.io_sleepedf import pairs, load, FS, CHANS
from ampint.spec import clean_runs, run_power, noise_fraction, peak_freq, aperiodic_fit
from ampint.summ import summaries
from ampint.chaos import surrogate_indices

OUT = "results/nf_sleep"
BANDS = {"alpha": ((7.0, 14.0), 2.0), "sigma250": ((10.5, 16.0), 1.5)}
CHUNK = (60.0, 112.0)  # matches the bank length prior (60-115 s)


def chunks(lab, M, stage):
    """Contiguous same-stage, artefact-free stretches split into 60-300 s chunks."""
    ok = (lab == stage) & M
    out = []
    for a, b in clean_runs(ok, int(CHUNK[0] * FS)):
        L = b - a
        nchunk = max(1, int(np.ceil(L / (CHUNK[1] * FS))))
        edges = np.linspace(a, b, nchunk + 1).astype(int)
        out += [(edges[i], edges[i + 1]) for i in range(nchunk) if edges[i + 1] - edges[i] >= CHUNK[0] * FS]
    return out


def one(psg, hyp):
    rec = os.path.basename(psg)[:8]
    of = f"{OUT}/{rec}.csv"
    if os.path.exists(of):
        return
    try:
        X, lab, M = load(psg, hyp)
        rows = []
        rng = np.random.default_rng(int(rec[3:6]))
        for ci, ch in enumerate(CHANS):
            for stage in ["W", "N1", "N2", "N3", "REM"]:
                for k, (a, b) in enumerate(chunks(lab, M[ci], stage)):
                    x = X[ci, a:b]
                    f, S = welch(x - x.mean(), FS, nperseg=int(4 * FS), noverlap=int(2 * FS))
                    base = dict(rec=rec, ch=ch, stage=stage, chunk=k, t_start_h=a / FS / 3600, dur=(b - a) / FS)
                    ap_b, ap_chi = aperiodic_fit(f, S)
                    base.update(ap_offset=ap_b, ap_exp=ap_chi)
                    for bname, (search, hw) in BANDS.items():
                        pk, pkh = peak_freq(f, S, search)
                        lo, hi = pk - hw, pk + hw
                        rho = noise_fraction(f, S, lo, hi, FS)
                        runs = run_power(x, np.ones(x.size, bool), lo, hi, FS, min_len_s=30.0, trim_s=1.0)
                        s = summaries(runs, rho, FS)
                        rows.append(dict(base, band=bname, kind="data", peak=pk, peak_height=pkh, **s))
                        if k % 4 == 0:  # surrogate summaries on every 4th chunk
                            sr = run_power(x, np.ones(x.size, bool), lo, hi, FS, min_len_s=30.0, surrogate_rng=rng)
                            rows.append(dict(base, band=bname, kind="iaaft0", peak=pk, peak_height=pkh, **summaries(sr, rho, FS)))
                    # chaos indices on every 4th chunk: first 20 s, decimated to 125 Hz as in TDBRAIN
                    if k % 4 == 1:
                        seg = x[:int(20 * FS)] - x[:int(20 * FS)].mean()
                        ci_ = surrogate_indices(decimate(seg, 2, zero_phase=True), FS / 2, n_surr=10, seed=k)
                        rows.append(dict(base, band="broadband", kind="chaos", **ci_))
        pd.DataFrame(rows).to_csv(of, index=False)
    except Exception:
        with open(f"{OUT}/errors.log", "a") as fh:
            fh.write(psg + "\n" + traceback.format_exc() + "\n")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    pp = pairs()
    if len(sys.argv) > 2:
        pp = pp[:int(sys.argv[2])]
    print(len(pp), "recordings", flush=True)
    Parallel(n_jobs=int(sys.argv[1]), verbose=5)(delayed(one)(a, b) for a, b in pp)


"""Linear twins: convergence of the IAAFT iteration and stability of the twin baseline.
(1) For clean runs of 40 healthy eyes-closed TDBRAIN recordings (O1, Oz, O2, Pz; up to three runs per channel), the
    relative error of the surrogate amplitude spectrum after the final rank-ordering step, the relative error of
    alpha-band power (peak +- 2 Hz) at 10, 50, 100 and 1,000 iterations, and the change of the twin's band-power
    summaries between 100 and 1,000 iterations relative to the spread between independent twins.
(2) Group-level stability: results with the first versus the second independent twin of each TDBRAIN channel.
usage: python scripts/twin_checks.py <n_jobs>"""
import sys, glob, json, numpy as np, pandas as pd
from joblib import Parallel, delayed
from ampint.io_tdbrain import load_rest, load_meta, ROOT, FS
from ampint.spec import clean_runs, analytic, noise_fraction, peak_freq
from ampint.pipeline import clean_psd
from ampint.summ import summaries
from ampint.nf import fold_delta

CH = ["O1", "Oz", "O2", "Pz"]
REC = (10, 50, 100, 1000)


def iaaft_track(x, rng, n_iter=1000):
    xs = np.sort(x); amp = np.abs(np.fft.rfft(x)); y = rng.permutation(x); out = {}
    for k in range(1, n_iter + 1):
        Y = np.fft.rfft(y)
        y = np.fft.irfft(amp * np.exp(1j * np.angle(Y)), x.size)
        y = xs[np.argsort(np.argsort(y))]
        if k in REC:
            out[k] = (float(np.linalg.norm(np.abs(np.fft.rfft(y)) - amp) / np.linalg.norm(amp)), y.copy())
    return out


def one(path, seed):
    X, M, badch, picks = load_rest(path)
    rows = []
    rng = np.random.default_rng(seed)
    for ci, ch in enumerate(picks):
        if ch not in CH or badch[ci]:
            continue
        f, S = clean_psd(X[ci], M[ci], FS)
        if f is None:
            continue
        pk, _ = peak_freq(f, S, (7.0, 14.0)); lo, hi = pk - 2.0, pk + 2.0
        rho = noise_fraction(f, S, lo, hi, FS)
        for ri, (a, b) in enumerate(clean_runs(M[ci], int(8 * FS))[:3]):
            seg = X[ci][a:b] - X[ci][a:b].mean()
            pb = np.mean(np.abs(analytic(seg, lo, hi, FS)[250:-250]) ** 2)
            tr = iaaft_track(seg, rng)
            alt = iaaft_track(seg, np.random.default_rng(seed + 10_000 + 7 * ci + ri), 100)[100][1]
            r = dict(path=path, ch=ch, run=ri, n=seg.size)
            for k, (e, y) in tr.items():
                r[f"spec_err_{k}"] = e
                r[f"band_err_{k}"] = float(abs(np.mean(np.abs(analytic(y, lo, hi, FS)[250:-250]) ** 2) / pb - 1))
            s100 = summaries([np.abs(analytic(tr[100][1], lo, hi, FS)[250:-250]) ** 2], rho, FS)
            s1000 = summaries([np.abs(analytic(tr[1000][1], lo, hi, FS)[250:-250]) ** 2], rho, FS)
            salt = summaries([np.abs(analytic(alt, lo, hi, FS)[250:-250]) ** 2], rho, FS)
            for k in ("cv", "lq99", "acf0.5", "mu_n", "c_n"):
                r[f"{k}_100"], r[f"{k}_1000"], r[f"{k}_alt"] = s100[k], s1000[k], salt[k]
            rows.append(r)
    return rows


def main():
    meta = load_meta(); m1 = meta[(meta.sessID == 1) & (meta.indication == "HEALTHY") & (meta.age >= 18)].drop_duplicates("sub")
    files = sorted(glob.glob(f"{ROOT}/sub-*/ses-1/eeg/*task-restEC*_eeg.bdf"))
    files = [f for f in files if f.split("/")[-1].split("_")[0] in set(m1["sub"])]
    files = list(np.random.default_rng(3).choice(files, 40, replace=False))
    R = pd.DataFrame([r for rr in Parallel(n_jobs=int(sys.argv[1]))(delayed(one)(p, 100 + i) for i, p in enumerate(files)) for r in rr])
    out = dict(n_recordings=int(R.path.nunique()), n_runs=len(R))
    out["spectral error (median, 95th pct)"] = {k: [float(R[f"spec_err_{k}"].median()), float(R[f"spec_err_{k}"].quantile(.95))] for k in REC}
    out["alpha-band power error (median, 95th pct)"] = {k: [float(R[f"band_err_{k}"].median()), float(R[f"band_err_{k}"].quantile(.95))] for k in REC}
    out["summary change 100->1000 iterations relative to twin-to-twin spread"] = {
        k: float(np.median(np.abs(R[f"{k}_1000"] - R[f"{k}_100"])) / np.median(np.abs(R[f"{k}_alt"] - R[f"{k}_100"]))) for k in ("cv", "lq99", "acf0.5", "mu_n", "c_n")}
    # (2) first vs second twin at group level
    p = pd.read_csv("results/post_tdbrain.csv", usecols=["sub", "ses", "task", "ch", "kind", "th_mu_q50", "th_c_q50"])
    p = p[p.ses == "ses-1"].merge(meta[meta.sessID == 1].drop_duplicates("sub")[["sub", "indication", "age"]], on="sub")
    p = p[(p.indication == "HEALTHY") & (p.age >= 18)]; p["delta"] = fold_delta(p.th_mu_q50, p.th_c_q50)
    g = p[p.kind.isin(["data", "iaaft0", "iaaft1"])].groupby(["sub", "task", "kind"]).delta.mean().unstack(["kind"])
    tw = {}
    for t in ("restEC", "restEO"):
        q = g.xs(t, level="task").dropna()
        tw[t] = dict(n=len(q), median_twin0=float(q.iaaft0.median()), median_twin1=float(q.iaaft1.median()),
                     spearman_twin0_twin1=float(q[["iaaft0", "iaaft1"]].corr("spearman").iloc[0, 1]),
                     eeg_minus_twin0=float((q.data - q.iaaft0).median()), eeg_minus_twin1=float((q.data - q.iaaft1).median()),
                     frac_eeg_above_twin0=float((q.data > q.iaaft0).mean()), frac_eeg_above_twin1=float((q.data > q.iaaft1).mean()))
    out["first vs second twin (healthy, O1/Oz/O2/Pz mean)"] = tw
    json.dump(out, open("results/twin_checks.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

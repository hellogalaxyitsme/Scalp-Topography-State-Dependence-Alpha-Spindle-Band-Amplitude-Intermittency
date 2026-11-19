"""Effect of fragmentation into artifact-free runs on the alpha coordinates and on the classifier score.
The simulation banks use contiguous records of the retained duration, whereas TDBRAIN recordings are analyzed in
artifact-free runs of at least 8 s. Here simulated signals (single source, bank priors; constant parameters with
additive noise, and modulation of mu with the prior of the modulated class) are processed three ways:
  cont  - contiguous record with the retained duration of the eyes-closed mask (as in the banks);
  EC    - the same signal cut by the artifact mask of an eyes-closed TDBRAIN recording;
  EO    - the same signal cut by the eyes-open mask of the same participant and channel.
Masks come from the healthy reference sample (first session, both conditions). Half of the simulations use masks of
Fz, Cz, Pz or Oz ("typical"), half use the most fragmented masks, eyes-closed channel recordings with at least three
runs on any channel ("fragmented"). Each version is
paired with its own IAAFT twin (per run) and passed through the alpha posterior and the main classifier.
EC - cont isolates fragmentation at fixed parameters; EO - EC isolates the condition dependence of the observation
process at fixed parameters.
usage: python scripts/obs_fragmentation.py masks <n_jobs>   -> results/obs_frag_masks.json
       python scripts/obs_fragmentation.py sims <n_jobs>    -> results/obs_fragmentation.csv
       python scripts/obs_fragmentation.py summary          -> results/obs_fragmentation.json"""
import sys, os, glob, json, joblib, numpy as np, pandas as pd
from joblib import Parallel, delayed
from sbi_bank import CONFIGS, PRIOR, MOD_PRIORS
from sbi_twinbank import signal
from sbi_fit import predict
from mech_twin import DIFF, CTX, NEAR
from ampint.nf import fold_delta
from ampint.spec import run_power, noise_fraction, peak_freq
from ampint.pipeline import clean_psd
from ampint.summ import summaries

CHANS = ["Fz", "Cz", "Pz", "Oz"]
MIN_RUN = 8.0
N_PER_CLASS = 800
OFFSET = 81_000_000
CFG = CONFIGS["alpha"]


def runs_of(m, fs):
    m = np.r_[False, m, False]; d = np.diff(m.astype(int))
    a, b = np.where(d == 1)[0], np.where(d == -1)[0]
    keep = (b - a) >= int(MIN_RUN * fs)
    return [[int(x), int(y)] for x, y in zip(a[keep], b[keep])]


def mask_one(sub):
    from ampint.io_tdbrain import load_rest, ROOT, FS
    out = {}
    for task in ("restEC", "restEO"):
        f = glob.glob(f"{ROOT}/{sub}/ses-1/eeg/{sub}_ses-1_task-{task}_eeg.bdf")
        if not f:
            return None
        X, M, bad, picks = load_rest(f[0])
        out[task] = {ch: dict(n=int(M.shape[1]), runs=runs_of(M[i], FS)) for i, ch in enumerate(picks) if not bad[i]}
    return sub, out


def masks(nj):
    R = pd.read_csv("results/nf_tdbrain_reclevel.csv")
    h = R[R.group == "HEALTHY"].pivot_table(index="sub", columns="task", values="th_mu_q50").dropna()
    res = Parallel(n_jobs=nj, verbose=5)(delayed(mask_one)(s) for s in sorted(h.index))
    json.dump({s: o for r in res if r for s, o in [r]}, open("results/obs_frag_masks.json", "w"))


def mask_array(spec):
    m = np.zeros(spec["n"], bool)
    for a, b in spec["runs"]:
        m[a:b] = True
    return m


def version(x, m, fs, rng):
    f, S = clean_psd(x, m, fs, MIN_RUN)
    pk, _ = peak_freq(f, S, CFG["search"])
    lo, hi = pk - CFG["hw"], pk + CFG["hw"]
    rho = noise_fraction(f, S, lo, hi, fs)
    s = summaries(run_power(x, m, lo, hi, fs, MIN_RUN), rho, fs)
    st = summaries(run_power(x, m, lo, hi, fs, MIN_RUN, surrogate_rng=rng), rho, fs)
    row = dict(s, peak=pk)
    row.update({"tw_" + k: v for k, v in st.items()})
    return row


def sim_one(k, cls, sub, ch, mk, sset):
    seed = OFFSET + (0 if cls == "const" else 1_000_000) + k
    rng = np.random.default_rng(seed)
    fs = CFG["fs"]
    ec, eo = mk["restEC"][ch], mk["restEO"][ch]
    n = max(ec["n"], eo["n"])
    th = dict(mu=rng.uniform(*PRIOR["mu"]), c=rng.uniform(*PRIOR["c"]), t0=10 ** rng.uniform(*PRIOR["log10_t0"]),
              chi=rng.uniform(*PRIOR["chi"]), snr_db=rng.uniform(*PRIOR["snr_db"]), f0=rng.uniform(*CFG["f0"]),
              T=n / fs - 2.0 + 0.05)
    mod = None
    if cls == "mod":
        mp = MOD_PRIORS["fastmod"]
        mod = dict(sd=rng.uniform(*mp["sd"]), tau=10 ** rng.uniform(*mp["log10_tau"]))
    x = signal(CFG, th, seed, mod, 1, 0.0)[:n]
    m_ec, m_eo = mask_array(ec), mask_array(eo)
    ret = sum(b - a - 2 * int(fs) for a, b in ec["runs"])
    if ret < 40 * fs or sum(b - a - 2 * int(fs) for a, b in eo["runs"]) < 40 * fs:
        return []
    m_ct = np.zeros(n, bool); m_ct[:ret + 2 * int(fs)] = True
    base = dict(k=k, cls=cls, set=sset, sub=sub, ch=ch, seed=seed, **{"th_" + a: b for a, b in th.items()},
                mod_sd=mod["sd"] if mod else 0.0, mod_tau=mod["tau"] if mod else np.nan,
                runs_ec=len(ec["runs"]), runs_eo=len(eo["runs"]))
    rows = []
    for v, m in (("cont", m_ct), ("EC", m_ec), ("EO", m_eo)):
        rows.append(dict(base, version=v, **version(x, m, fs, np.random.default_rng(seed + 999))))
    return rows


def sims(nj):
    mk = json.load(open("results/obs_frag_masks.json"))
    both = [(s, ch) for s, o in mk.items() for ch in o["restEC"] if ch in o["restEO"]]
    sets = {"typical": [p for p in both if p[1] in CHANS],
            "fragmented": [p for p in both if len(mk[p[0]]["restEC"][p[1]]["runs"]) >= 3]}
    print({k: len(v) for k, v in sets.items()}, flush=True)
    rng = np.random.default_rng(2026)
    jobs = []
    for cls in ("const", "mod"):
        k = 0
        for sset, pairs in sets.items():
            for i in rng.integers(0, len(pairs), N_PER_CLASS // 2):
                jobs.append((k, cls, *pairs[i], sset)); k += 1
    res = Parallel(n_jobs=nj, verbose=5)(delayed(sim_one)(k, cls, s, ch, mk[s], ss) for k, cls, s, ch, ss in jobs)
    pd.DataFrame([r for rr in res for r in rr]).to_csv("results/obs_fragmentation.csv", index=False)


def boot_mean(d, n=2000, seed=0):
    r = np.random.default_rng(seed); d = np.asarray(d, float)
    b = [d[r.integers(0, d.size, d.size)].mean() for _ in range(n)]
    return [float(d.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def summary():
    D = pd.read_csv("results/obs_fragmentation.csv").replace([np.inf, -np.inf], np.nan)
    reg = joblib.load("results/sbi_alpha_full.joblib")
    D = D[D[reg["features"]].notna().all(1)].reset_index(drop=True)
    pr = predict(reg, D)
    D["mu_hat"], D["c_hat"], D["t0_hat"] = pr.th_mu_q50.values, pr.th_c_q50.values, pr.log_t0_q50.values
    D["d_hat"] = fold_delta(D.mu_hat.values, D.c_hat.values)
    D["d_true"] = fold_delta(D.th_mu.values, D.th_c.values)
    C = joblib.load("results/mechtwb_alpha.joblib")
    for f in DIFF:
        D["d_" + f] = D[f] - D["tw_" + f]
    ok = D[C["cols"]].notna().all(1)
    D["score"] = np.nan
    D.loc[ok, "score"] = C["iso"].predict(C["clf"].predict_proba(D.loc[ok, C["cols"]].values)[:, 1])
    W = D.pivot_table(index=["k", "cls"], columns="version", values=["d_hat", "mu_hat", "c_hat", "t0_hat", "score"])
    W.columns = [f"{a}_{b}" for a, b in W.columns]
    W = W.dropna().reset_index()
    info = D[D.version == "EC"][["k", "cls", "set", "d_true", "runs_ec", "runs_eo", "T", "sub", "ch"]]
    W = W.merge(info, on=["k", "cls"], how="inner")
    out = dict(n=int(len(W)), n_by_class={c: int((W.cls == c).sum()) for c in ("const", "mod")})
    out["runs"] = {t: {str(int(a)): int(b) for a, b in zip(*np.unique(np.clip(W[t].values, 1, 5), return_counts=True))}
                   for t in ("runs_ec", "runs_eo")}
    out["retained_s_ec_median"] = {s: float(W[W["set"] == s]["T"].median()) for s in ("typical", "fragmented")}
    mk = json.load(open("results/obs_frag_masks.json"))
    nr = {t: np.array([len(v["runs"]) for o in mk.values() for v in o[t].values()]) for t in ("restEC", "restEO")}
    out["empirical runs per channel recording (healthy, 26 channels)"] = {
        t: dict(n=int(a.size), one=float((a == 1).mean()), two=float((a == 2).mean()), three_plus=float((a >= 3).mean()), zero=float((a == 0).mean()))
        for t, a in nr.items()}
    sp = lambda a, b: float(pd.Series(np.asarray(a)).corr(pd.Series(np.asarray(b)), method="spearman"))
    res = {}
    groups = {"const": W.cls == "const", "mod": W.cls == "mod", "all": W.cls.notna(),
              "typical": W["set"] == "typical", "fragmented": W["set"] == "fragmented",
              "const, fragmented": (W.cls == "const") & (W["set"] == "fragmented")}
    for cls, gm in groups.items():
        w = W[gm]
        r = {}
        for q in ("d_hat", "mu_hat", "c_hat", "t0_hat", "score"):
            r[q] = {"EC-cont": boot_mean(w[q + "_EC"] - w[q + "_cont"]), "EO-EC": boot_mean(w[q + "_EO"] - w[q + "_EC"]),
                    "spearman EC vs cont": sp(w[q + "_EC"], w[q + "_cont"]),
                    "sd EC-cont": float((w[q + "_EC"] - w[q + "_cont"]).std())}
        multi = w[w.runs_ec >= 2]
        r["d_hat, EC-cont, recordings with >=2 runs"] = boot_mean(multi.d_hat_EC - multi.d_hat_cont) if len(multi) > 10 else None
        r["n with >=2 runs"] = int(len(multi))
        near = w[w.d_hat_cont.between(*NEAR)]
        r["score, near fold (cont estimate): EC-cont"] = boot_mean(near.score_EC - near.score_cont) if len(near) > 10 else None
        r["score, near fold: mean cont, EC"] = [float(near.score_cont.mean()), float(near.score_EC.mean())] if len(near) else None
        r["n near fold"] = int(len(near))
        if cls.startswith("const"):
            for v in ("cont", "EC"):
                e = w["d_hat_" + v] - w["d_true"]
                r[f"recovery {v}"] = dict(r2=float(1 - (e ** 2).sum() / ((w["d_true"] - w["d_true"].mean()) ** 2).sum()),
                                         spearman=sp(w["d_hat_" + v], w["d_true"]),
                                         median_bias=float(e.median()), mae=float(e.abs().mean()))
        res[cls] = r
    out["results"] = res
    json.dump(out, open("results/obs_fragmentation.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"masks": lambda: masks(int(sys.argv[2])), "sims": lambda: sims(int(sys.argv[2])), "summary": summary}[sys.argv[1]]()

"""What the model coordinates add beyond direct temporal descriptors of the amplitude.
Direct descriptors of the band power series (each also referenced to the recording's linear twin): amplitude
coefficient of variation, kurtosis of normalised power, 99th percentile of log normalised power (upper tail) and the
power-autocorrelation timescale (first lag below 1/e). Model coordinates: fold distance delta, criticality c and
amplitude timescale log10 t0 (posterior medians). Endpoints:
(1) state sensitivity: paired effect size of eye opening (TDBRAIN healthy adults, mean of O1, Oz, O2, Pz) and of
    N2 minus wakefulness (ANPHY-Sleep scalp mean; Sleep-EDF Fpz-Cz), with participant-bootstrap intervals;
(2) test-retest reliability over two TDBRAIN sessions (ICC(2,1), absolute agreement);
(3) cross-session prediction of session-2 burst statistics: Spearman correlations of the session-1 model
    prediction, session-1 EEG and session-1 twin with bootstrap intervals, and the contribution of the model
    prediction in a rank regression that also contains the session-1 EEG statistic, the twin and the direct
    descriptors.
usage: python scripts/descriptor_benchmark.py"""
import glob, json, numpy as np, pandas as pd
from scipy.stats import spearmanr
from ampint.io_tdbrain import load_meta
from ampint.nf import fold_delta
from ampint.summ import DLAGS

POST4 = {"O1", "OZ", "O2", "PZ"}
DESC = ["cv", "kurt", "lq99", "tau_e"]
MODEL = ["delta", "th_c_q50", "log_t0_q50"]
STATS = ["b2_frac", "b_frac", "b_dur_mean", "b_dur_p90", "b2_dur_mean", "b_rate"]
NB = 2000
rng = np.random.default_rng(0)


def add_measures(d):
    a = d[[f"a{i:02d}" for i in range(40)]].values
    ll = np.log(DLAGS)
    tau = np.full(len(d), np.nan)
    for i, row in enumerate(a):
        k = np.where(row < np.exp(-1))[0]
        if len(k) and k[0] > 0:
            j = k[0]; w = (row[j - 1] - np.exp(-1)) / (row[j - 1] - row[j])
            tau[i] = np.exp(ll[j - 1] + w * (ll[j] - ll[j - 1]))
        elif len(k) == 0:
            tau[i] = DLAGS[-1]
    d["tau_e"] = np.log10(tau)
    d["delta"] = fold_delta(d.th_mu_q50.values, d.th_c_q50.values)
    return d


def boot_ci(f, n, nb=NB):
    v = np.array([f(rng.integers(0, n, n)) for _ in range(nb)])
    return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]


def dz(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(x.mean() / x.std(ddof=1))


def icc21(a, b):
    """ICC(2,1), two-way random effects, absolute agreement (Shrout & Fleiss 1979)."""
    Y = np.column_stack([a, b]); Y = Y[np.isfinite(Y).all(1)]
    n, k = Y.shape; gm = Y.mean()
    msr = k * np.sum((Y.mean(1) - gm) ** 2) / (n - 1); msc = n * np.sum((Y.mean(0) - gm) ** 2) / (k - 1)
    mse = np.sum((Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + gm) ** 2) / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


def tdbrain_table():
    cols = ["sub", "ses", "task", "ch", "kind", "cv", "kurt", "lq99", "th_mu_q50", "th_c_q50", "log_t0_q50"] + [f"a{i:02d}" for i in range(40)]
    p = pd.read_csv("results/post_tdbrain.csv", usecols=cols)
    p = add_measures(p[p.ch.str.upper().isin(POST4) & p.kind.isin(["data", "iaaft0"])].copy())
    k = ["sub", "ses", "task", "ch"]
    D = p[p.kind == "data"].set_index(k)[DESC + MODEL]
    T = p[p.kind == "iaaft0"].set_index(k)[DESC + MODEL]
    R = D.join(T, rsuffix="_tw", how="inner")
    for m in DESC + MODEL:
        R[m + "_ref"] = R[m] - R[m + "_tw"]
    return R.groupby(level=["sub", "ses", "task"]).mean()


def main():
    out = {}
    R = tdbrain_table()
    meas = DESC + MODEL + [m + "_ref" for m in DESC + MODEL]
    meta = load_meta(); m1 = meta[meta.sessID == 1].drop_duplicates("sub")[["sub", "indication", "age"]]
    # (1) eye opening, healthy adults, first session
    H = R.xs("ses-1", level="ses").reset_index().merge(m1, on="sub")
    H = H[(H.indication == "HEALTHY") & (H.age >= 18)]
    W = H.pivot_table(index="sub", columns="task", values=meas).dropna()
    eo = {}
    for m in meas:
        x = (W[(m, "restEO")] - W[(m, "restEC")]).values
        eo[m] = dict(n=len(x), dz=dz(x), ci=boot_ci(lambda i: dz(x[i]), len(x)))
    out["EO-EC effect size (healthy, O1/Oz/O2/Pz)"] = eo
    # (2) test-retest, participants with two sessions (TDBRAIN, all indications)
    rel = {}
    for task in ("restEC", "restEO"):
        S = R.xs(task, level="task").reset_index().pivot_table(index="sub", columns="ses", values=meas)
        rel[task] = {}
        for m in meas:
            if (m, "ses-2") not in S:
                continue
            ab = S[[(m, "ses-1"), (m, "ses-2")]].dropna().values
            rel[task][m] = dict(n=len(ab), icc=icc21(ab[:, 0], ab[:, 1]), rho=float(spearmanr(ab[:, 0], ab[:, 1])[0]),
                                icc_ci=boot_ci(lambda i: icc21(ab[i, 0], ab[i, 1]), len(ab)))
    out["test-retest ICC(2,1) (two TDBRAIN sessions)"] = rel
    # (3) cross-session burst prediction with direct descriptors as competitors
    b = pd.concat([pd.read_csv(f) for f in glob.glob("results/bursts_retest/*.csv") if open(f).readline().strip()])
    key = ["sub", "task", "ch"]
    obs2 = b[(b.kind == "data") & (b.ses == "ses-2")].set_index(key)[STATS]
    obs1 = b[(b.kind == "data") & (b.ses == "ses-1")].set_index(key)[STATS]
    tw1 = b[(b.kind == "twin") & (b.ses == "ses-1")].set_index(key)[STATS]
    pred = pd.read_csv("results/bursts_ppc_retest.csv").groupby(key)[STATS].mean()
    idx = obs2.index.intersection(obs1.index).intersection(tw1.index).intersection(pred.index)
    L = pd.concat({"obs2": obs2.loc[idx], "obs1": obs1.loc[idx], "twin1": tw1.loc[idx], "pred1": pred.loc[idx]}, axis=1)
    L = L.groupby(level=["sub", "task"]).mean()
    L.columns = [f"{a}|{b}" for a, b in L.columns]
    S1 = R.xs("ses-1", level="ses")
    cs = {}
    for task in ("restEC", "restEO"):
        r = L.xs(task, level="task").join(S1.xs(task, level="task")[DESC + [m + "_ref" for m in DESC] + ["delta"]], how="inner")
        for s in STATS:
            q = r[[f"{a}|{s}" for a in ("obs2", "obs1", "twin1", "pred1")] + DESC + [m + "_ref" for m in DESC] + ["delta"]].dropna()
            o2, o1, tw, pr = (q[f"{a}|{s}"].rank().values for a in ("obs2", "obs1", "twin1", "pred1"))
            n = len(q)
            rho = lambda x, y, i: spearmanr(x[i], y[i])[0]
            e = dict(n=n)
            for name, v in (("pred1", pr), ("obs1", o1), ("twin1", tw)):
                e[f"rho_{name}"] = float(spearmanr(o2, v)[0]); e[f"rho_{name}_ci"] = boot_ci(lambda i: rho(o2, v, i), n)
            e["rho_pred1_minus_obs1_ci"] = boot_ci(lambda i: rho(o2, pr, i) - rho(o2, o1, i), n)
            e["rho_desc"] = {m: float(spearmanr(o2, q[m].values)[0]) for m in DESC + [m + "_ref" for m in DESC] + ["delta"]}
            # rank regression: does the model prediction add to session-1 EEG, twin and direct descriptors?
            Z = np.column_stack([np.ones(n), o1, tw] + [q[m].rank().values for m in DESC + [m + "_ref" for m in DESC]] + [pr])
            beta, *_ = np.linalg.lstsq(Z, o2, rcond=None)
            res = o2 - Z @ beta; sig2 = res @ res / (n - Z.shape[1])
            e["t_pred1_beyond_obs1_twin_desc"] = float(beta[-1] / np.sqrt(sig2 * np.linalg.inv(Z.T @ Z)[-1, -1]))
            Z0 = np.column_stack([np.ones(n), tw] + [q[m].rank().values for m in DESC + [m + "_ref" for m in DESC]] + [pr])
            beta0, *_ = np.linalg.lstsq(Z0, o2, rcond=None)
            res0 = o2 - Z0 @ beta0; sig20 = res0 @ res0 / (n - Z0.shape[1])
            e["t_pred1_beyond_twin_desc"] = float(beta0[-1] / np.sqrt(sig20 * np.linalg.inv(Z0.T @ Z0)[-1, -1]))
            cs[f"{task}|{s}"] = e
    out["cross-session prediction"] = cs
    # (1b) N2 - wake in the sleep data
    sl = {}
    for t, unit, chf in (("anphy_sigma", "rec", None), ("sleep_sigma", "rec", "EEG Fpz-Cz")):
        cols = ["rec", "ch", "stage", "chunk", "kind", "cv", "kurt", "lq99", "th_mu_q50", "th_c_q50", "log_t0_q50"] + [f"a{i:02d}" for i in range(40)]
        P = pd.read_csv(f"results/post_{t}.csv", usecols=cols)
        if chf:
            P = P[P.ch == chf]
        else:
            P = P[~P.ch.str.contains("-")]
        P = add_measures(P[P.stage.isin(["W", "N2"])].copy())
        if t == "sleep_sigma":
            P["rec"] = P.rec.str[:5]   # subject (nights pooled at the piece level)
        k = ["rec", "ch", "stage", "chunk"]
        Dd = P[P.kind == "data"].set_index(k)[DESC + MODEL]; Tt = P[P.kind == "iaaft0"].set_index(k)[DESC + MODEL]
        J = Dd.join(Tt, rsuffix="_tw", how="left")
        for m in DESC + MODEL:
            J[m + "_ref"] = J[m] - J[m + "_tw"]
        G = J.groupby(level=["rec", "stage"]).mean().reset_index().pivot_table(index="rec", columns="stage", values=meas)
        sl[t] = {}
        for m in meas:
            x = (G[(m, "N2")] - G[(m, "W")]).dropna().values
            sl[t][m] = dict(n=len(x), dz=dz(x), ci=boot_ci(lambda i: dz(x[i]), len(x)))
    out["N2-W effect size"] = sl
    json.dump(out, open("results/descriptor_benchmark.json", "w"), indent=1)
    pd.set_option("display.width", 250)
    print(pd.DataFrame({m: dict(dz=v["dz"], lo=v["ci"][0], hi=v["ci"][1], n=v["n"]) for m, v in eo.items()}).T.round(2).to_string())
    for task in rel:
        print(task, pd.DataFrame(rel[task]).T[["n", "icc", "rho"]].round(2).to_string())
    print(pd.DataFrame({k: {a: (v[a] if not isinstance(v[a], list) else str(np.round(v[a], 2))) for a in
                            ["n", "rho_pred1", "rho_obs1", "rho_twin1", "rho_pred1_minus_obs1_ci", "t_pred1_beyond_obs1_twin_desc", "t_pred1_beyond_twin_desc"]}
                        for k, v in cs.items()}).T.to_string())
    for t in sl:
        print(t, pd.DataFrame({m: dict(dz=v["dz"], lo=v["ci"][0], hi=v["ci"][1], n=v["n"]) for m, v in sl[t].items()}).T.round(2).to_string())


if __name__ == "__main__":
    main()

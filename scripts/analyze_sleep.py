"""Sleep-EDF: normal-form coordinates (alpha, sigma) and chaos indices across wake and sleep stages.
inputs: results/post_sleep_alpha.csv, results/post_sleep_sigma.csv (nf_apply outputs), results/nf_sleep/*.csv"""
import glob, json, numpy as np, pandas as pd
import statsmodels.formula.api as smf

STAGES = ["W", "N1", "N2", "N3", "REM"]
NF = ["th_mu_q50", "th_c_q50", "log_t0_q50", "p_focus", "p_bistable", "p_oscillatory", "slow_flag", "slow_excess", "cv", "rho", "peak"]
IDX = ["lle", "pe", "sampen", "lzc", "hfd"]


def subj_id(rec):
    return rec[:5]  # SC4ss -> subject


def mixed(d, y, n_boot=2000, seed=0):
    """Within-participant stage effects vs wake: per-participant difference (stage - W) of stage means,
    mean difference with bootstrap 95% CI, paired d_z and Wilcoxon signed-rank p (robust; no model fitting)."""
    from scipy.stats import wilcoxon
    w = d.pivot_table(index="subj", columns="stage", values=y)
    rng = np.random.default_rng(seed); res = {}
    for st in STAGES[1:]:
        if st not in w or "W" not in w:
            continue
        dd = (w[st] - w["W"]).dropna().values
        if len(dd) < 10:
            continue
        bs = [rng.choice(dd, dd.size).mean() for _ in range(n_boot)]
        res[st] = dict(coef=float(dd.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
                       dz=float(dd.mean() / (dd.std(ddof=1) + 1e-12)), n=int(dd.size), p=float(wilcoxon(dd).pvalue))
    return res


def main():
    out = {}
    for band, fn in [("alpha", "results/post_sleep_alpha.csv"), ("sigma", "results/post_sleep_sigma.csv")]:
        p = pd.read_csv(fn)
        p["subj"] = p.rec.map(subj_id)
        dat = p[p.kind == "data"]
        # chunk -> subject x stage x channel means
        s = dat.groupby(["subj", "ch", "stage"])[NF].mean().reset_index()
        out[band] = {"median_by_stage": {ch: s[s.ch == ch].groupby("stage")[NF].median().reindex(STAGES).round(3).to_dict(orient="index")
                                         for ch in s.ch.unique()},
                     "n_chunks_by_stage": dat.groupby(["ch", "stage"]).size().unstack().to_dict(orient="index")}
        reg = dat.assign(r=dat[["p_focus", "p_bistable", "p_oscillatory"]].values.argmax(1))
        out[band]["regime_frac"] = {ch: reg[reg.ch == ch].groupby("stage").r.value_counts(normalize=True).unstack().reindex(STAGES).round(3).to_dict(orient="index")
                                    for ch in s.ch.unique()}
        out[band]["mixed"] = {}
        for ch in s.ch.unique():
            for y in ["th_mu_q50", "th_c_q50", "log_t0_q50", "p_bistable", "slow_excess", "cv"]:
                try:
                    out[band]["mixed"][f"{ch}|{y}"] = mixed(s[s.ch == ch], y)
                except Exception as e:
                    out[band]["mixed"][f"{ch}|{y}"] = str(e)
        sur = p[p.kind == "iaaft0"]
        out[band]["surrogate_median_by_stage"] = sur.groupby("stage")[["th_mu_q50", "th_c_q50", "p_bistable", "slow_flag", "cv"]].median().reindex(STAGES).round(3).to_dict(orient="index")
        s.to_csv(f"results/sleep_{band}_subject_stage.csv", index=False)
    # chaos indices by stage: raw vs twin vs surrogate-z
    raw = pd.concat([pd.read_csv(f) for f in glob.glob("results/nf_sleep/*.csv") if open(f).readline().strip()], ignore_index=True)
    ch = raw[raw.kind == "chaos"].copy(); ch["subj"] = ch.rec.map(subj_id)
    cs = ch.groupby(["subj", "ch", "stage"])[[*IDX, *[k + "_sm" for k in IDX], *[k + "_z" for k in IDX]]].mean().reset_index()
    out["chaos_median_by_stage"] = cs.groupby("stage")[[*IDX, *[k + "_sm" for k in IDX], *[k + "_z" for k in IDX]]].median().reindex(STAGES).round(3).to_dict(orient="index")
    out["chaos_mixed"] = {}
    for k in IDX:
        for col in [k, k + "_sm", k + "_z"]:
            try:
                out["chaos_mixed"][col] = mixed(cs, col)
            except Exception as e:
                out["chaos_mixed"][col] = str(e)
    cs.to_csv("results/sleep_chaos_subject_stage.csv", index=False)
    json.dump(out, open("results/sleep_summary.json", "w"), indent=1, default=str)
    print(json.dumps({k: out[k] for k in ["chaos_median_by_stage"]}, indent=1)[:3000])
    for band in ["alpha", "sigma"]:
        print("==", band)
        for ch_, v in out[band]["median_by_stage"].items():
            print(ch_); print(pd.DataFrame(v).T[["th_mu_q50", "th_c_q50", "log_t0_q50", "p_bistable", "slow_flag", "cv", "rho"]])


if __name__ == "__main__":
    main()

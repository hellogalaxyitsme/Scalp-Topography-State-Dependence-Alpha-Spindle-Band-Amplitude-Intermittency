"""Third cohort (ds003478): depressive symptom severity (BDI) vs normal-form coordinates and chaos indices."""
import glob, json, numpy as np, pandas as pd
from scipy.stats import spearmanr
from analyze_mumtaz import eff
from ampint.paths import DS003478_ROOT

POST_CH = ["O1", "OZ", "O2", "PZ", "P3", "P4", "POZ", "PO3", "PO4"]
NF = ["th_mu_q50", "th_c_q50", "log_t0_q50", "p_bistable", "slow_excess", "cv", "peak", "rho"]
IDX = ["lle", "pe", "sampen", "lzc", "hfd"]


def partial_spearman(x, y, z):
    from scipy.stats import rankdata
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    rx, ry, rz = rankdata(x[ok]), rankdata(y[ok]), rankdata(z[ok])
    res = lambda a: a - np.polyval(np.polyfit(rz, a, 1), rz)
    r = np.corrcoef(res(rx), res(ry))[0, 1]
    n = ok.sum(); t = r * np.sqrt((n - 3) / (1 - r * r))
    from scipy.stats import t as tdist
    return float(r), float(2 * tdist.sf(abs(t), n - 3)), int(n)


def main():
    meta = pd.read_csv(f"{DS003478_ROOT}/participants.tsv", sep="\t").rename(columns={"participant_id": "sub"})
    post = pd.read_csv("results/post_ds003478.csv")
    post = post[post.ch.isin(POST_CH)]
    s = post.groupby(["sub", "task"])[NF].mean().reset_index().merge(meta[["sub", "sex", "age", "BDI"]], on="sub")
    raw = pd.concat([pd.read_csv(f) for f in glob.glob("results/ds003478/*.csv") if open(f).readline().strip()], ignore_index=True)
    ch = raw[raw.kind == "chaos"].groupby(["sub", "task"])[[*IDX, *[k + "_sm" for k in IDX], *[k + "_z" for k in IDX]]].mean().reset_index()
    ch = ch.merge(meta[["sub", "sex", "BDI"]], on="sub")
    out = {"n": int(s["sub"].nunique()), "nf_bdi": {}, "nf_group": {}, "chaos_bdi": {}, "chaos_group": {}}
    for task in ["restEC", "restEO"]:
        t = s[s.task == task]
        hi, lo = t[t.BDI >= 13], t[t.BDI <= 6]
        for k in NF:
            r, p, n = partial_spearman(t[k].values, t.BDI.values.astype(float), t.sex.values.astype(float))
            out["nf_bdi"][f"{task}|{k}"] = dict(rho=r, p=p, n=n)
            out["nf_group"][f"{task}|{k}"] = eff(hi[k], lo[k])
        c = ch[ch.task == task]
        for k in IDX:
            for col in [k, k + "_sm", k + "_z"]:
                r, p, n = partial_spearman(c[col].values, c.BDI.values.astype(float), c.sex.values.astype(float))
                out["chaos_bdi"][f"{task}|{col}"] = dict(rho=r, p=p, n=n)
                out["chaos_group"][f"{task}|{col}"] = eff(c[c.BDI >= 13][col], c[c.BDI <= 6][col])
    json.dump(out, open("results/ds003478_summary.json", "w"), indent=1)
    for sec in ["nf_bdi", "nf_group", "chaos_bdi", "chaos_group"]:
        print("==", sec)
        for k, v in out[sec].items():
            if "rho" in v:
                print(f"{k:28s} rho={v['rho']:+.2f} p={v['p']:.4f} n={v['n']}")
            else:
                print(f"{k:28s} d={v['d']:+.2f} p={v['p']:.4f} n={v['n_mdd']}/{v['n_hc']}")


if __name__ == "__main__":
    main()

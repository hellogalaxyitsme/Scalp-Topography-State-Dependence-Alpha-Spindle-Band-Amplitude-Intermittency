"""External MDD replication (Mumtaz): chaos indices and normal-form coordinates, MDD vs healthy.
input: results/mumtaz/*.csv and results/post_mumtaz.csv (posterior on kind==nf rows)."""
import glob, json, numpy as np, pandas as pd

IDX = ["lle", "pe", "sampen", "lzc", "hfd"]
POST_CH = ["O1", "O2", "P3", "P4", "Pz"]
NF = ["th_mu_q50", "th_c_q50", "log_t0_q50", "p_bistable", "slow_excess", "cv", "peak", "rho"]


def eff(x, y, n_perm=10000, seed=0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    sp = np.sqrt(((len(x) - 1) * x.var(ddof=1) + (len(y) - 1) * y.var(ddof=1)) / (len(x) + len(y) - 2))
    d = (x.mean() - y.mean()) / sp
    rng = np.random.default_rng(seed); a = np.r_[x, y]; c = 0
    for _ in range(n_perm):
        p = rng.permutation(a); c += abs(p[:len(x)].mean() - p[len(x):].mean()) >= abs(x.mean() - y.mean())
    return dict(n_mdd=len(x), n_hc=len(y), d=float(d), p=float((c + 1) / (n_perm + 1)))


def main():
    d = pd.concat([pd.read_csv(f) for f in glob.glob("results/mumtaz/*.csv") if open(f).readline().strip()], ignore_index=True)
    ch = d[d.kind == "chaos"]
    out = {"chaos": {}, "nf": {}}
    subj = ch.groupby(["group", "sub", "task"])[[*IDX, *[k + "_sm" for k in IDX], *[k + "_z" for k in IDX]]].mean().reset_index()
    for task in ["restEC", "restEO"]:
        s = subj[subj.task == task]
        for k in IDX:
            for col, kind in [(k, "raw"), (k + "_sm", "surrogate_twin"), (k + "_z", "surrogate_z")]:
                out["chaos"][f"{task}|{k}|{kind}"] = eff(s[s.group == "MDD"][col], s[s.group == "HEALTHY"][col])
    post = pd.read_csv("results/post_mumtaz.csv")
    post = post[post.ch.isin(POST_CH)]
    ps = post.groupby(["group", "sub", "task"])[NF].mean().reset_index()
    for task in ["restEC", "restEO"]:
        s = ps[ps.task == task]
        for k in NF:
            out["nf"][f"{task}|{k}"] = eff(s[s.group == "MDD"][k], s[s.group == "HEALTHY"][k])
    json.dump(out, open("results/mumtaz_summary.json", "w"), indent=1)
    for sec in out:
        print("==", sec)
        for k, v in out[sec].items():
            print(f"{k:40s} d={v['d']:+.2f} p={v['p']:.4f} n={v['n_mdd']}/{v['n_hc']}")


if __name__ == "__main__":
    main()

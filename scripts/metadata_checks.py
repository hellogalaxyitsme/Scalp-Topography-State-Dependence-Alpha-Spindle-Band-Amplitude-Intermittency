"""TDBRAIN metadata checks: (1) interval between the two sessions of the test-retest sample (difference of the ages
recorded per session) and treatment records (rTMS or neurofeedback protocol, post-treatment scores); test-retest
ICC(2,1) of the Table-2 measures for all participants with two sessions, those without treatment records and the
healthy participants; (2) the MDD contrast along the state axis for participants whose formal_status confirms MDD,
against all participants referred with an MDD indication (age/sex model fitted to healthy participants, axis of
state_axis_joint.py, participant bootstrap of both groups).
usage: python scripts/metadata_checks.py  -> results/metadata_checks.json"""
import json, numpy as np, pandas as pd
from ampint.io_tdbrain import load_meta
from descriptor_benchmark import tdbrain_table, icc21, DESC, MODEL

NB = 2000
rng = np.random.default_rng(4)
MC = ["th_mu_q50", "th_c_q50"]


def ci(f, n):
    v = np.array([f(rng.integers(0, n, n)) for _ in range(NB)])
    return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]


def main():
    meta = load_meta()
    out = {}
    R = tdbrain_table()
    S2 = set(R.xs("ses-2", level="ses").index.get_level_values("sub"))
    S1 = set(R.xs("ses-1", level="ses").index.get_level_values("sub"))
    rt = sorted(S1 & S2)
    M = meta[meta["sub"].isin(rt)].copy()
    M["treated"] = (M["rTMS PROTOCOL"].notna() | M["NF Protocol"].notna() | M["BDI_post"].notna() | M["YBOCS_post"].notna()
                    | M["ADHD_post_Att_leading"].notna() | M["ADHD_post_Hyp_leading"].notna())
    g = M.groupby("sub").agg(indication=("indication", "first"), treated=("treated", "any"))
    ages = M.pivot_table(index="sub", columns="sessID", values="age", aggfunc="first")
    gap = (ages[2] - ages[1]).dropna() if 1 in ages and 2 in ages else pd.Series(dtype=float)
    out["retest sample"] = dict(n=len(rt), indication=g.indication.fillna("unknown").value_counts().to_dict(),
                                with_treatment_records=int(g.treated.sum()),
                                treatment_records_by_indication=g[g.treated].indication.value_counts().to_dict(),
                                interval_years=dict(n_with_both_ages=int(len(gap)), median=float(gap.median()),
                                                    q25=float(gap.quantile(.25)), q75=float(gap.quantile(.75)),
                                                    n_negative=int((gap < 0).sum())))
    meas = MODEL + DESC
    subsets = {"all": rt, "without treatment records": sorted(set(rt) - set(g.index[g.treated])),
               "healthy": sorted(g.index[g.indication == "HEALTHY"])}
    icc = {}
    for name, subs in subsets.items():
        icc[name] = {}
        for task in ("restEC", "restEO"):
            T = R.xs(task, level="task").reset_index()
            T = T[T["sub"].isin(subs)].pivot_table(index="sub", columns="ses", values=meas)
            icc[name][task] = {}
            for m in meas:
                ab = T[[(m, "ses-1"), (m, "ses-2")]].dropna().values
                icc[name][task][m] = dict(n=int(len(ab)), icc=icc21(ab[:, 0], ab[:, 1]), ci=ci(lambda i: icc21(ab[i, 0], ab[i, 1]), len(ab)))
    out["test-retest ICC(2,1) by subset"] = icc
    A = json.load(open("results/state_axis_joint.json"))["axis"]
    u = np.array([A["u_mu"], A["u_c"]])
    rl = pd.read_csv("results/nf_tdbrain_reclevel.csv")
    ec = rl[rl.task == "restEC"].dropna(subset=["age", "gender"] + MC).copy()
    fs = meta[meta.sessID == 1].drop_duplicates("sub")[["sub", "formal_status"]]
    ec = ec.merge(fs, on="sub", how="left")
    a = (ec.age.values - 45) / 15
    X = np.column_stack([np.ones(len(ec)), a, a * a, ec.gender.values.astype(float)])
    ref = (ec.group == "HEALTHY").values
    beta = np.linalg.lstsq(X[ref], ec[MC].values[ref], rcond=None)[0]
    res = (ec[MC].values - X @ beta) @ u
    mdd = {}
    for name, m in (("indication MDD", (ec.group == "MDD").values), ("indication and formal_status MDD", ((ec.group == "MDD") & (ec.formal_status == "MDD")).values),
                    ("indication MDD, formal_status unknown", ((ec.group == "MDD") & (ec.formal_status == "UNKNOWN")).values)):
        x, h = res[m], res[ref]
        est = x.mean() - h.mean()
        bs = np.array([rng.choice(x, len(x)).mean() - rng.choice(h, len(h)).mean() for _ in range(NB)])
        mdd[name] = dict(n=int(m.sum()), n_healthy=int(ref.sum()), along=float(est), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                         p=float(min(1.0, max(1 / NB, 2 * min(np.mean(bs <= 0), np.mean(bs >= 0))))))
    out["MDD along the state axis by diagnostic field"] = mdd
    json.dump(out, open("results/metadata_checks.json", "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in out.items() if k != "test-retest ICC(2,1) by subset"}, indent=1, default=str))
    for name in icc:
        print(name, {t: {m: round(v["icc"], 2) for m, v in icc[name][t].items()} for t in icc[name]}, {t: next(iter(icc[name][t].values()))["n"] for t in icc[name]})


if __name__ == "__main__":
    main()

"""Direct fit of the stationary power density as a baseline for the simulation-based estimates. The closed-form
maximum-likelihood coordinates of the power density, fitted to the band power without (mu_n, c_n) and with (mu_d, c_d)
deconvolution of the aperiodic background, are part of the summary statistics of every recording and simulation.
They are compared with the posterior medians on
(1) held-out simulations: recovery of delta and c over the whole prior and in the healthy-alpha region, and the
    dependence of the estimation error on the signal-to-background ratio;
(2) empirical endpoints as in descriptor_benchmark.py: paired effect size of eye opening (TDBRAIN healthy adults,
    O1/Oz/O2/Pz) and of N2 minus wakefulness (ANPHY-Sleep scalp mean, Sleep-EDF Fpz-Cz), test-retest ICC(2,1),
    the EEG-twin difference with eyes closed, and the rank correlation of the eye-opening change with the change in
    the background fraction;
(3) twin-referenced test-retest reliability with one twin and with the mean of the two twins of each channel.
usage: python scripts/direct_fit_baseline.py  -> results/direct_fit_baseline.json"""
import json, numpy as np, pandas as pd
from scipy.stats import spearmanr
from ampint.io_tdbrain import load_meta
from ampint.nf import fold_delta
from descriptor_benchmark import dz, icc21, boot_ci, POST4

EST = {"SBI": ("th_mu_q50", "th_c_q50"), "direct": ("mu_n", "c_n"), "direct deconvolved": ("mu_d", "c_d")}
MEAS = [f"{q}|{e}" for q in ("delta", "c") for e in EST]
HEALTHY_BOX = ((-7.0, -3.0), (1.5, 5.5))


def add_est(d):
    for e, (m, c) in EST.items():
        d[f"delta|{e}"] = fold_delta(d[m].values, d[c].values)
        d[f"c|{e}"] = d[c].values
    return d


def recovery(H):
    true = {"delta": fold_delta(H.th_mu.values, H.th_c.values), "c": H.th_c.values}
    box = H.th_mu.between(*HEALTHY_BOX[0]) & H.th_c.between(*HEALTHY_BOX[1])
    out = {}
    for region, m in (("whole prior", np.ones(len(H), bool)), ("healthy region", box.values)):
        out[region] = {}
        for q in ("delta", "c"):
            for e in EST:
                y, x = true[q][m], H[f"{q}|{e}"].values[m]
                ok = np.isfinite(x)
                y, x, snr = y[ok], x[ok], H.th_snr_db.values[m][ok]
                err = x - y
                out[region][f"{q}|{e}"] = dict(n=int(ok.sum()), r2=float(1 - np.mean(err ** 2) / np.var(y)),
                                               rs=float(spearmanr(x, y)[0]), bias=float(err.mean()),
                                               mae=float(np.mean(np.abs(err))), rs_error_snr=float(spearmanr(err, snr)[0]))
    return out


def tdbrain():
    cols = ["sub", "ses", "task", "ch", "kind", "rho", "mu_n", "c_n", "mu_d", "c_d", "th_mu_q50", "th_c_q50", "log_t0_q50"]
    p = pd.read_csv("results/post_tdbrain.csv", usecols=cols)
    p = add_est(p[p.ch.str.upper().isin(POST4) & p.kind.isin(["data", "iaaft0", "iaaft1"])].copy())
    p["delta|SBI"] = fold_delta(p.th_mu_q50.values, p.th_c_q50.values)
    p["log_t0"] = p.log_t0_q50
    return p.groupby(["sub", "ses", "task", "kind"])[MEAS + ["rho", "log_t0"]].mean()


def main():
    out = {"recovery": {}}
    for cfg in ("alpha", "sigma250"):
        H = add_est(pd.read_csv(f"results/sbi_{cfg}_full_heldout.csv"))
        out["recovery"][cfg] = recovery(H)
    R = tdbrain()
    meta = load_meta(); m1 = meta[meta.sessID == 1].drop_duplicates("sub")[["sub", "indication", "age"]]
    D = R.xs("data", level="kind")
    # eye opening, healthy adults, first session
    H = D.xs("ses-1", level="ses").reset_index().merge(m1, on="sub")
    H = H[(H.indication == "HEALTHY") & (H.age >= 18)]
    W = H.pivot_table(index="sub", columns="task", values=MEAS + ["rho"]).dropna()
    drho = (W[("rho", "restEO")] - W[("rho", "restEC")]).values
    eo = {}
    for m in MEAS:
        x = (W[(m, "restEO")] - W[(m, "restEC")]).values
        eo[m] = dict(n=len(x), dz=dz(x), ci=boot_ci(lambda i: dz(x[i]), len(x)), rs_with_delta_rho=float(spearmanr(x, drho)[0]))
    out["EO-EC (healthy, O1/Oz/O2/Pz)"] = eo
    # EEG minus twin, eyes closed, healthy adults
    T = R.xs("iaaft0", level="kind").xs("ses-1", level="ses").reset_index().merge(m1, on="sub")
    T = T[(T.indication == "HEALTHY") & (T.age >= 18) & (T.task == "restEC")].set_index("sub")
    E = H[H.task == "restEC"].set_index("sub")
    tw = {}
    for m in MEAS:
        x = (E[m] - T[m]).dropna().values
        tw[m] = dict(n=len(x), median=float(np.median(x)), dz=dz(x), frac_positive=float(np.mean(x > 0)))
    out["EEG-twin, eyes closed (healthy)"] = tw
    # test-retest
    rel = {}
    for task in ("restEC", "restEO"):
        S = D.xs(task, level="task").reset_index().pivot_table(index="sub", columns="ses", values=MEAS)
        rel[task] = {}
        for m in MEAS:
            ab = S[[(m, "ses-1"), (m, "ses-2")]].dropna().values
            rel[task][m] = dict(n=len(ab), icc=icc21(ab[:, 0], ab[:, 1]), icc_ci=boot_ci(lambda i: icc21(ab[i, 0], ab[i, 1]), len(ab)))
    out["test-retest ICC(2,1)"] = rel
    # twin-referenced reliability: one twin versus the mean of two twins
    if "iaaft1" in R.index.get_level_values("kind"):
        T0, T1 = R.xs("iaaft0", level="kind"), R.xs("iaaft1", level="kind")
        trel = {}
        for task in ("restEC", "restEO"):
            trel[task] = {}
            for m in ("delta|SBI", "c|SBI", "log_t0"):
                one = (D[m] - T0[m]).xs(task, level="task")
                two = (D[m] - (T0[m] + T1[m]) / 2).xs(task, level="task")
                tt = (T0[m] - T1[m]).xs(task, level="task")
                r = {}
                for lab, v in (("one twin", one), ("two twins", two)):
                    S = v.reset_index().pivot_table(index="sub", columns="ses", values=m)
                    ab = S[["ses-1", "ses-2"]].dropna().values
                    r[lab] = dict(n=len(ab), icc=icc21(ab[:, 0], ab[:, 1]))
                r["single-twin noise variance / variance of the twin-referenced measure"] = float(np.nanvar(tt.values) / 2 / np.nanvar(one.values))
                trel[task][m] = r
        out["twin-referenced ICC"] = trel
    # N2 - wake in the sleep data
    sl = {}
    for t, chf in (("anphy_sigma", None), ("sleep_sigma", "EEG Fpz-Cz")):
        cols = ["rec", "ch", "stage", "chunk", "kind", "mu_n", "c_n", "mu_d", "c_d", "th_mu_q50", "th_c_q50"]
        P = pd.read_csv(f"results/post_{t}.csv", usecols=cols)
        P = P[(P.ch == chf) if chf else ~P.ch.str.contains("-")]
        P = add_est(P[P.stage.isin(["W", "N2"]) & (P.kind == "data")].copy())
        if t == "sleep_sigma":
            P["rec"] = P.rec.str[:5]
        G = P.groupby(["rec", "stage"])[MEAS].mean().reset_index().pivot_table(index="rec", columns="stage", values=MEAS)
        sl[t] = {}
        for m in MEAS:
            x = (G[(m, "N2")] - G[(m, "W")]).dropna().values
            sl[t][m] = dict(n=len(x), dz=dz(x), ci=boot_ci(lambda i: dz(x[i]), len(x)))
    out["N2-W"] = sl
    json.dump(out, open("results/direct_fit_baseline.json", "w"), indent=1)
    pd.set_option("display.width", 250)
    for cfg, r in out["recovery"].items():
        for region, v in r.items():
            print(cfg, region); print(pd.DataFrame(v).T.round(3).to_string())
    print("EO-EC"); print(pd.DataFrame({m: dict(dz=v["dz"], lo=v["ci"][0], hi=v["ci"][1], rs_drho=v["rs_with_delta_rho"]) for m, v in eo.items()}).T.round(2).to_string())
    print("EEG-twin EC"); print(pd.DataFrame(tw).T.round(2).to_string())
    for task in rel:
        print(task, pd.DataFrame({m: dict(icc=v["icc"], lo=v["icc_ci"][0], hi=v["icc_ci"][1], n=v["n"]) for m, v in rel[task].items()}).T.round(2).to_string())
    if "twin-referenced ICC" in out:
        print(json.dumps(out["twin-referenced ICC"], indent=1))
    for t in sl:
        print(t); print(pd.DataFrame({m: dict(dz=v["dz"], lo=v["ci"][0], hi=v["ci"][1], n=v["n"]) for m, v in sl[t].items()}).T.round(2).to_string())


if __name__ == "__main__":
    main()

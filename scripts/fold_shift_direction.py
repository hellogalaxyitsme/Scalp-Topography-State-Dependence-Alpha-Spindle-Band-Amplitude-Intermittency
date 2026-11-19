"""Direction of the apparent fold shift and finite-record limit (extends fold_bias_theory.py).
(1) Exact moment-matching shift of the fold coordinate, Delta delta = delta(mu*, c*) - delta(mu0, c), over the
    noisy-focus region for s = 0.5 and 1, compared with the leading-order law
    Delta delta = s^2 (c/2 - <u>_0) for c > 0 and -s^2 <u>_0 for c <= 0 (moments at the baseline).
(2) Simulations of the normal form with Ornstein-Uhlenbeck modulation of mu (mu0 = -5, c = 0.5, s = 2) for records of
    T = 3000 and 300 t0 and correlation times up to T, fitted by moment matching per record.
usage: python scripts/fold_shift_direction.py -> results/fold_shift_direction.json, fold_shift_direction_map.csv,
       fold_shift_direction_sim.csv"""
import json, os, numpy as np, pandas as pd
from fold_bias_theory import U, moments, fit, delta, closed_form, simulate

XH, WH = np.polynomial.hermite_e.hermegauss(48)
WH = WH / WH.sum()


def mix_moments(mu0, c, s):
    M = np.array([moments(mu0 + s * x, c) for x in XH])
    return float(WH @ M[:, 0]), float(WH @ M[:, 1])


def shift(mu0, c, s):
    m1, m2 = mix_moments(mu0, c, s)
    p, ok = fit(m1, m2, closed_form(mu0, c, s))
    u1, _ = moments(mu0, c)
    lead = s ** 2 * (c / 2 - u1) if c > 0 else -s ** 2 * u1
    return dict(mu0=mu0, c=c, s=s, mean_u=u1, mu_star=p[0], c_star=p[1], d0=delta(mu0, c), d_star=delta(*p),
                dd=delta(*p) - delta(mu0, c), dd_lead=lead, ok=bool(ok))


def main():
    os.makedirs("results", exist_ok=True)
    rows = []
    for c in np.round(np.arange(-1.0, 5.51, 0.1), 2):
        for mu0 in np.round(np.arange(-8.0, -0.09, 0.1), 2):
            if mu0 >= (-c * c / 4 if c > 0 else 0.0) - 0.05:
                continue
            for s in (0.5, 1.0):
                rows.append(shift(mu0, c, s))
    Mp = pd.DataFrame(rows)
    Mp.to_csv("results/fold_shift_direction_map.csv", index=False)
    out = {}
    for s in (0.5, 1.0):
        m = Mp[(Mp.s == s) & Mp.ok]
        pos = m.c > 0
        hr = m.mu0.between(-7, -3) & m.c.between(1.5, 5.5)
        agree = np.sign(m.dd) == np.sign(m.dd_lead)
        out[f"s={s}"] = dict(n=int(len(m)), frac_toward_all=float((m.dd > 0).mean()),
                             frac_toward_c_pos=float((m[pos].dd > 0).mean()),
                             frac_toward_healthy_region=float((m[hr].dd > 0).mean()),
                             min_dd_healthy_region=float(m[hr].dd.min()),
                             sign_agreement_with_leading_order=float(agree.mean()),
                             max_abs_err_leading_order=float((m.dd - m.dd_lead).abs().max()),
                             median_rel_err_leading_order_c_pos=float(((m[pos].dd - m[pos].dd_lead).abs() / m[pos].dd.abs().clip(lower=1e-3)).median()))
    ex = {f"{a},{b},{s}": shift(a, b, s) for a, b, s in [(-1, 0.5, 0.5), (-5, 0.1, 0.5), (-5, 0.5, 2.0), (-4, 0.5, 1.0), (-3, 0.3, 1.0)]}
    out["examples"] = {k: {kk: (float(v) if not isinstance(v, bool) else v) for kk, v in d.items()} for k, d in ex.items()}
    print(json.dumps(out, indent=1))
    mu0, c, s = -5.0, 0.5, 2.0
    sims = []
    for T, taus, nrep in ((3000.0, [100.0, 300.0], 24), (300.0, [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0], 48)):
        t, m1, m2 = simulate(mu0, c, s, taus, nrep=nrep, T=T, seed=7)
        for tau, a, b in zip(t, m1, m2):
            p, ok = fit(a, b, closed_form(mu0, c, s))
            sims.append(dict(T=T, tau=tau, mu_fit=p[0], c_fit=p[1], delta_fit=delta(*p), ok=bool(ok)))
    S = pd.DataFrame(sims)
    S.to_csv("results/fold_shift_direction_sim.csv", index=False)
    g = S.groupby(["T", "tau"]).delta_fit.agg(["mean", "std", "count"])
    print(g.round(3).to_string())
    out["finite_record"] = {f"T={a:g},tau={b:g}": dict(mean=float(r["mean"]), sd=float(r["std"]), n=int(r["count"])) for (a, b), r in g.iterrows()}
    json.dump(out, open("results/fold_shift_direction.json", "w"), indent=1)


if __name__ == "__main__":
    main()

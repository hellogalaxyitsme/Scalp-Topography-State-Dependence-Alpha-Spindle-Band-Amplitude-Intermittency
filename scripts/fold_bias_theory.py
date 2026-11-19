"""Apparent fold proximity under slow fluctuations of the bifurcation parameter.

Stationary power density of the scale-free stochastic Bautin normal form: p(u) ~ exp(mu u + c u^2/2 - u^3/3), u >= 0.
If mu fluctuates as mu0 + s*xi (xi standard normal) slowly compared with the amplitude timescale, the power density is
the mixture over xi. The constant-parameter density closest to it in Kullback-Leibler divergence (the limit of a
maximum-likelihood fit to a long record) matches E[u] and E[u^2]. Second-order closed form:
c_eff = c + s_e^2, mu_eff = mu0 - s_e^2 <u>, s_e^2 = s^2 / (1 + s^2 Var u), moments at (mu0, c).
Part 1 compares exact moment matching with the closed form; part 2 simulates the normal form with Ornstein-Uhlenbeck
modulation of mu (correlation time tau, in units of t0) and fits the power density by moment matching.
usage: python scripts/fold_bias_theory.py  -> results/fold_bias_theory.json, results/fold_bias_theory_sim.csv"""
import json, os, numpy as np, pandas as pd
from scipy.optimize import fsolve

U = np.linspace(0, 12, 6001)


def moments(mu, c):
    f = mu * U + 0.5 * c * U ** 2 - U ** 3 / 3
    w = np.exp(f - f.max()); w /= np.trapezoid(w, U)
    m1 = np.trapezoid(U * w, U); m2 = np.trapezoid(U ** 2 * w, U)
    return m1, m2


def mixture_moments(mu0, c, s, n=401):
    x = np.linspace(-6, 6, n); g = np.exp(-x ** 2 / 2); g /= g.sum()
    M = np.array([moments(mu0 + s * xi, c) for xi in x])
    return float(g @ M[:, 0]), float(g @ M[:, 1])


def fit(m1, m2, start):
    sol = fsolve(lambda p: np.subtract(moments(*p), (m1, m2)), start, full_output=True)
    return sol[0], sol[2] == 1


def delta(mu, c):
    return mu + c ** 2 / 4 if c > 0 else mu


def closed_form(mu0, c, s):
    m1, m2 = moments(mu0, c); v = m2 - m1 ** 2
    se2 = s ** 2 / (1 + s ** 2 * v)
    return mu0 - se2 * m1, c + se2


def simulate(mu0, c, s, taus, nrep=24, T=3000.0, dt=0.004, seed=1):
    rng = np.random.default_rng(seed)
    taus = np.repeat(np.asarray(taus, float), nrep)
    n = len(taus); z = np.full(n, 0.3 + 0j); xi = rng.standard_normal(n)
    a = np.exp(-dt / taus); b = np.sqrt(1 - a ** 2)
    nstep = int(T / dt); burn = int(min(20 * max(1.0, taus.max()), T / 5) / dt)
    S1 = np.zeros(n); S2 = np.zeros(n); cnt = 0
    for k in range(nstep + burn):
        xi = a * xi + b * rng.standard_normal(n)
        mu = mu0 + s * xi
        u = (z * z.conjugate()).real
        z = z + (mu + c * u - u ** 2) * z * dt + np.sqrt(dt) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
        if k >= burn and k % 5 == 0:
            u = (z * z.conjugate()).real; S1 += u; S2 += u ** 2; cnt += 1
    return taus, S1 / cnt, S2 / cnt


def main():
    os.makedirs("results", exist_ok=True)
    rows = []
    for mu0 in (-7.0, -5.5, -4.0):
        for c in (-1.0, 0.5, 2.0):
            for s in (1.0, 2.0, 3.0):
                m1, m2 = mixture_moments(mu0, c, s)
                cf = closed_form(mu0, c, s)
                ex, ok = fit(m1, m2, cf)
                rows.append(dict(mu0=mu0, c=c, s=s, delta0=delta(mu0, c), mu_exact=ex[0], c_exact=ex[1], ok=bool(ok),
                                 delta_exact=delta(*ex), mu_cf=cf[0], c_cf=cf[1], delta_cf=delta(*cf)))
    R = pd.DataFrame(rows)
    print(R.round(2).to_string())
    taus = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
    mu0, c, s = -5.0, 0.5, 2.0
    t, m1, m2 = simulate(mu0, c, s, taus)
    sim = []
    for tau, a, b in zip(t, m1, m2):
        p, ok = fit(a, b, closed_form(mu0, c, s))
        sim.append(dict(tau=tau, mu_fit=p[0], c_fit=p[1], delta_fit=delta(*p), ok=bool(ok)))
    S = pd.DataFrame(sim)
    g = S.groupby("tau").agg(['mean', 'std'])
    print(g.round(3).to_string())
    qs = mixture_moments(mu0, c, s); ex, _ = fit(*qs, closed_form(mu0, c, s))
    out = dict(grid=rows, sim_params=dict(mu0=mu0, c=c, s=s), quasi_static_exact=dict(mu=ex[0], c=ex[1], delta=delta(*ex)),
               closed_form=dict(zip(("mu", "c"), closed_form(mu0, c, s))), constant=dict(mu=mu0, c=c, delta=delta(mu0, c)))
    json.dump(out, open("results/fold_bias_theory.json", "w"), indent=1)
    S.to_csv("results/fold_bias_theory_sim.csv", index=False)


if __name__ == "__main__":
    main()

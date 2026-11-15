"""How the fold distance delta relates to direct intermittency statistics across the simulated parameter space.
For the alpha bank (true parameters) and the stationary density of Eq. (2): Spearman correlation of delta with the
coefficient of variation of power, the upper-tail probability P(P > 5 E[P]) (exact, from the density) and, from the
simulated band power, amplitude CV, upper log-power quantile and power-autocorrelation timescale; over the whole
prior, for c <= 0 and c > 0 separately, and over the empirically occupied region (posterior medians of healthy
TDBRAIN alpha, central 90%). Partial correlations control for log10 t0.
usage: python scripts/delta_relations.py"""
import json, numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata
from sbi_fit import load_bank
from ampint.nf import fold_delta
from descriptor_benchmark import add_measures

U = np.linspace(0, 40, 20001)


def exact(mu, c):
    lp = mu * U + 0.5 * c * U ** 2 - U ** 3 / 3
    p = np.exp(lp - lp.max()); p /= np.trapezoid(p, U)
    m = np.trapezoid(U * p, U); v = np.trapezoid((U - m) ** 2 * p, U)
    return np.sqrt(v) / m, np.trapezoid(np.where(U > 5 * m, p, 0), U)


def partial(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz); ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return float(np.corrcoef(ex, ey)[0, 1])


def main():
    b = load_bank("alpha")
    b = b.sample(min(len(b), 30000), random_state=0).reset_index(drop=True)
    b["delta_true"] = fold_delta(b.th_mu.values, b.th_c.values)
    ex = np.array([exact(m, c) for m, c in zip(b.th_mu.values, b.th_c.values)])
    b["cv_power_exact"], b["tail5_exact"] = ex[:, 0], ex[:, 1]
    b = add_measures(b.assign(th_mu_q50=b.th_mu, th_c_q50=b.th_c))
    post = pd.read_csv("results/post_tdbrain.csv", usecols=["kind", "ses", "th_mu_q50", "th_c_q50"])
    post = post[(post.kind == "data") & (post.ses == "ses-1")]
    lo_mu, hi_mu = post.th_mu_q50.quantile([.05, .95]); lo_c, hi_c = post.th_c_q50.quantile([.05, .95])
    regions = {"whole prior": np.ones(len(b), bool), "c <= 0": (b.th_c <= 0).values, "c > 0": (b.th_c > 0).values,
               f"empirical region (mu {lo_mu:.1f} to {hi_mu:.1f}, c {lo_c:.1f} to {hi_c:.1f})":
                   (b.th_mu.between(lo_mu, hi_mu) & b.th_c.between(lo_c, hi_c)).values}
    stats = ["cv_power_exact", "tail5_exact", "cv", "lq99", "tau_e"]
    out = {}
    for name, m in regions.items():
        s = b[m]
        out[name] = dict(n=int(m.sum()), spearman={k: float(spearmanr(s.delta_true, s[k], nan_policy="omit")[0]) for k in stats},
                         partial_given_log_t0={k: partial(s.delta_true.values, s[k].values, s.log_t0.values) for k in ["cv", "lq99", "tau_e"]},
                         spearman_c={k: float(spearmanr(s.th_c, s[k], nan_policy="omit")[0]) for k in stats})
    iso = b[(b.delta_true + 1).abs() < 0.25]
    out["iso-delta spread (delta = -1 +/- 0.25)"] = dict(n=len(iso), cv_power_exact_range=[float(iso.cv_power_exact.quantile(q)) for q in (.05, .95)],
                                                       tail5_exact_range=[float(iso.tail5_exact.quantile(q)) for q in (.05, .95)])
    json.dump(out, open("results/delta_relations.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

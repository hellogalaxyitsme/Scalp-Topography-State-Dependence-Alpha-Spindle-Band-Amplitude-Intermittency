"""Parameter-matched coverage percentile for every row of the posterior tables (see ood_matched.py).
Writes results/coverage_<name>.csv with one column (cov_pct) in the row order of the corresponding post file."""
import numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
from sbi_fit import load_bank
from ampint.summ import FEATURES
from ood_matched import PAR


def scorer(cfg):
    df = load_bank(cfg)
    tr = df[np.random.default_rng(0).random(len(df)) < 0.8]
    H = pd.read_csv(f"results/sbi_{cfg}_full_heldout.csv")
    mu, sd = tr[FEATURES].mean(), tr[FEATURES].std().replace(0, 1)
    Z = lambda d: ((d[FEATURES] - mu) / sd).values
    nn = NearestNeighbors(n_neighbors=10).fit(Z(tr.sample(min(40000, len(tr)), random_state=0)))
    H["knn"] = nn.kneighbors(Z(H))[0].mean(1)
    pm, ps = H[PAR].mean(), H[PAR].std()
    pnn = NearestNeighbors(n_neighbors=20).fit(((H[PAR] - pm) / ps).values)

    def score(e):
        out = np.full(len(e), np.nan)
        ok = e[FEATURES + PAR].replace([np.inf, -np.inf], np.nan).notna().all(1).values
        x = e[ok]
        for s in range(0, len(x), 20000):
            b = x.iloc[s:s + 20000]
            d_e = nn.kneighbors(Z(b))[0].mean(1)
            idx = pnn.kneighbors(((b[PAR] - pm) / ps).values)[1]
            out[np.where(ok)[0][s:s + len(b)]] = (H["knn"].values[idx] < d_e[:, None]).mean(1) * 100
        return out
    return score


if __name__ == "__main__":
    cols = FEATURES + PAR
    sa = scorer("alpha")
    for name in ["tdbrain", "mumtaz", "ds003478", "sleep_alpha"]:
        e = pd.read_csv(f"results/post_{name}.csv", usecols=cols)
        pd.DataFrame(dict(cov_pct=sa(e))).to_csv(f"results/coverage_{name}.csv", index=False)
        print(name, len(e), flush=True)
    ss = scorer("sigma250")
    e = pd.read_csv("results/post_sleep_sigma.csv", usecols=cols)
    pd.DataFrame(dict(cov_pct=ss(e))).to_csv("results/coverage_sleep_sigma.csv", index=False)
    print("sleep_sigma", len(e))

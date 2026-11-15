"""Normative modelling and group statistics."""
import numpy as np, pandas as pd
from sklearn.model_selection import KFold

GROUPS = ["MDD", "ADHD", "SMC", "OCD", "INSOMNIA", "TINNITUS", "PARKINSON"]


def design(df, extra=()):
    a = (df["age"].values - 45.0) / 15.0
    cols = [np.ones(len(df)), a, a ** 2, df["gender"].values.astype(float)]
    for e in extra:
        v = df[e].values.astype(float); cols.append((v - np.nanmean(v)) / (np.nanstd(v) + 1e-12))
    return np.column_stack(cols)


def normative_z(df, y, ref_mask, extra=(), n_splits=10, seed=0):
    """z = (y - yhat_ref(covariates)) / sd_ref. Reference (healthy) z computed out-of-fold."""
    X = design(df, extra); yv = df[y].values.astype(float)
    ok = np.isfinite(yv) & np.isfinite(X).all(1)
    z = np.full(len(df), np.nan)
    ref = np.where(ref_mask & ok)[0]; oth = np.where(~ref_mask & ok)[0]
    beta, *_ = np.linalg.lstsq(X[ref], yv[ref], rcond=None)
    sd = np.std(yv[ref] - X[ref] @ beta, ddof=X.shape[1])
    z[oth] = (yv[oth] - X[oth] @ beta) / sd
    for tr, te in KFold(n_splits, shuffle=True, random_state=seed).split(ref):
        b, *_ = np.linalg.lstsq(X[ref[tr]], yv[ref[tr]], rcond=None)
        s = np.std(yv[ref[tr]] - X[ref[tr]] @ b, ddof=X.shape[1])
        z[ref[te]] = (yv[ref[te]] - X[ref[te]] @ b) / s
    return z


def group_effects(df, zcol, group_col="group", ref="HEALTHY", groups=GROUPS, n_perm=5000, n_boot=2000, seed=0):
    """Mean normative z per group vs reference: difference, bootstrap 95% CI, permutation p (two-sided)."""
    rng = np.random.default_rng(seed)
    r = df.loc[df[group_col] == ref, zcol].dropna().values
    rows = []
    for g in groups:
        x = df.loc[df[group_col] == g, zcol].dropna().values
        if len(x) < 10:
            continue
        d = x.mean() - r.mean()
        bs = [rng.choice(x, x.size).mean() - rng.choice(r, r.size).mean() for _ in range(n_boot)]
        allv = np.r_[x, r]; cnt = 0
        for _ in range(n_perm):
            p = rng.permutation(allv)
            cnt += abs(p[:x.size].mean() - p[x.size:].mean()) >= abs(d)
        rows.append(dict(group=g, n=len(x), n_ref=len(r), diff=d, ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5),
                         p=(cnt + 1) / (n_perm + 1)))
    return pd.DataFrame(rows)


def fdr(p):
    p = np.asarray(p, float); n = p.size; o = np.argsort(p)
    q = np.empty(n); prev = 1.0
    for i in range(n - 1, -1, -1):
        prev = min(prev, p[o[i]] * n / (i + 1)); q[o[i]] = prev
    return q


def channel_adjacency(names, pos):
    """Channel neighbourhood from a Delaunay triangulation of the 2-D projected electrode positions.
    names: channel names; pos: dict name -> xyz. Returns a boolean adjacency matrix."""
    from scipy.spatial import Delaunay
    xyz = np.array([pos[n] for n in names], float)
    xy = project2d(xyz)
    A = np.zeros((len(names), len(names)), bool)
    for s in Delaunay(xy).simplices:
        for i in s:
            for j in s:
                A[i, j] = i != j
    # drop long edges along the convex hull
    d = np.linalg.norm(xyz[:, None] - xyz[None], axis=2)
    A &= d <= 2.0 * np.median(d[A])
    return A


def project2d(xyz):
    """Stereographic projection of electrode positions onto the plane, centred on the mean electrode
    direction from a least-squares sphere centre."""
    xyz = np.asarray(xyz, float)
    G = np.column_stack([2 * xyz, np.ones(len(xyz))])
    c = np.linalg.lstsq(G, np.sum(xyz ** 2, 1), rcond=None)[0][:3]
    u = xyz - c; u /= np.linalg.norm(u, axis=1, keepdims=True)
    v = u.mean(0); v /= np.linalg.norm(v)
    e1 = np.cross(v, [0, 0, 1.0] if abs(v[2]) < 0.9 else [1.0, 0, 0]); e1 /= np.linalg.norm(e1)
    e2 = np.cross(v, e1)
    z = u @ v
    return np.column_stack([u @ e2, u @ e1]) / (1 + z[:, None])


def _tstat(D):
    n = np.sum(np.isfinite(D), 0)
    return np.nanmean(D, 0) / (np.nanstd(D, 0, ddof=1) / np.sqrt(n)), n


def _welch(A, B):
    na, nb = np.sum(np.isfinite(A), 0), np.sum(np.isfinite(B), 0)
    va, vb = np.nanvar(A, 0, ddof=1) / na, np.nanvar(B, 0, ddof=1) / nb
    return (np.nanmean(A, 0) - np.nanmean(B, 0)) / np.sqrt(va + vb), na + nb


def _clusters(t, thr, adj):
    from scipy.sparse.csgraph import connected_components
    out = []
    for sgn in (1, -1):
        idx = np.where(sgn * t > thr)[0]
        if idx.size == 0:
            continue
        _, lab = connected_components(adj[np.ix_(idx, idx)], directed=False)
        for k in np.unique(lab):
            mem = idx[lab == k]
            out.append((sgn, mem, float(np.sum(np.abs(t[mem])))))
    return out


def cluster_test(D, adj, B=None, n_perm=5000, alpha_thr=0.05, seed=0):
    """Cluster-based permutation test over channels (Maris & Oostenveld 2007), NaN-tolerant.
    One-sample (paired differences D, subjects x channels; sign flips) or, with B, two-sample
    (D and B, subjects x channels; label shuffles; Welch t). Cluster-forming threshold: two-sided
    alpha_thr; statistic: cluster mass. Returns per-channel t and the clusters with permutation p."""
    from scipy.stats import t as tdist
    rng = np.random.default_rng(seed)
    if B is None:
        t, n = _tstat(D)
    else:
        t, n = _welch(D, B)
    thr = tdist.ppf(1 - alpha_thr / 2, max(2, int(np.median(n)) - 2))
    obs = _clusters(t, thr, adj)
    null = np.zeros(n_perm)
    if B is None:
        for i in range(n_perm):
            s = rng.choice([-1.0, 1.0], size=(D.shape[0], 1))
            tp, _ = _tstat(D * s)
            null[i] = max([m for _, _, m in _clusters(tp, thr, adj)], default=0.0)
    else:
        AB = np.vstack([D, B]); na = D.shape[0]
        for i in range(n_perm):
            p = rng.permutation(AB.shape[0])
            tp, _ = _welch(AB[p[:na]], AB[p[na:]])
            null[i] = max([m for _, _, m in _clusters(tp, thr, adj)], default=0.0)
    cl = [dict(sign=int(s), channels=mem.tolist(), mass=m, p=float((1 + np.sum(null >= m)) / (n_perm + 1)))
          for s, mem, m in obs]
    return t, sorted(cl, key=lambda c: c["p"])


def _intercept_t(V, x):
    """t statistic of the intercept in the regression of each row of V (replicates x subjects) on [1, x]."""
    Z = np.column_stack([np.ones(len(x)), x])
    G = np.linalg.inv(Z.T @ Z)
    a = (G @ Z.T)[0]                               # intercept = a @ y
    Q, _ = np.linalg.qr(Z)
    rss = np.sum(V ** 2, 1) - np.sum((V @ Q) ** 2, 1)
    return (V @ a) / np.sqrt(rss / (len(x) - 2) * G[0, 0])


def cluster_test_cov(D, X, adj, n_perm=5000, alpha_thr=0.05, seed=0):
    """Cluster-based test of the intercept in a per-channel regression of paired differences D on a covariate X
    (both subjects x channels, NaN-tolerant), i.e. the mean difference at zero covariate change. Permutation scheme
    of Freedman and Lane with sign flipping (Winkler et al. 2014): residuals of the reduced model D = b X + e are
    sign-flipped, the reduced fit is added back and the full model is refitted in every permutation; the same flips
    are applied to all channels, preserving their dependence. Returns per-channel t, intercept, slope and effect
    size (intercept / residual SD) and the clusters with permutation p."""
    from scipy.stats import t as tdist
    rng = np.random.default_rng(seed)
    nsub, nch = D.shape
    S = rng.choice([-1.0, 1.0], size=(n_perm, nsub))
    t_obs, tp = np.full(nch, np.nan), np.full((n_perm, nch), np.nan)
    icpt, slope, es, nn = (np.full(nch, np.nan) for _ in range(4))
    for j in range(nch):
        ok = np.isfinite(D[:, j]) & np.isfinite(X[:, j])
        d, x = D[ok, j], X[ok, j]
        b0 = (x @ d) / (x @ x)                     # reduced model (no intercept)
        e0 = d - b0 * x
        # the intercept and residuals depend on the data only through the reduced-model residuals
        t_obs[j] = _intercept_t(e0[None, :], x)[0]
        tp[:, j] = _intercept_t(S[:, ok] * e0[None, :], x)
        Z = np.column_stack([np.ones(len(x)), x]); coef = np.linalg.lstsq(Z, d, rcond=None)[0]
        icpt[j], slope[j], nn[j] = coef[0], coef[1], ok.sum()
        es[j] = coef[0] / np.sqrt(np.sum((d - Z @ coef) ** 2) / (len(x) - 2))
    thr = tdist.ppf(1 - alpha_thr / 2, max(2, int(np.median(nn)) - 2))
    obs = _clusters(t_obs, thr, adj)
    null = np.array([max([m for _, _, m in _clusters(tp[i], thr, adj)], default=0.0) for i in range(n_perm)])
    cl = [dict(sign=int(s), channels=mem.tolist(), mass=m, p=float((1 + np.sum(null >= m)) / (n_perm + 1)))
          for s, mem, m in obs]
    return dict(t=t_obs, intercept=icpt, slope=slope, es=es, n=nn), sorted(cl, key=lambda c: c["p"])


def cv_r2(X, y, n_splits=10, seed=0):
    """Out-of-fold R^2 of a linear model (with intercept)."""
    ok = np.isfinite(y) & np.isfinite(X).all(1); X, y = X[ok], y[ok]
    X1 = np.column_stack([np.ones(len(X)), X]); pred = np.empty(len(y))
    for tr, te in KFold(n_splits, shuffle=True, random_state=seed).split(X1):
        b, *_ = np.linalg.lstsq(X1[tr], y[tr], rcond=None); pred[te] = X1[te] @ b
    return 1 - np.mean((y - pred) ** 2) / np.var(y)

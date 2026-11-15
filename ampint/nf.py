"""Stochastic Bautin (generalised Hopf) normal-form inference from oscillation power.

Model (complex amplitude z, isotropic additive noise of intensity sigma):
    dz = [(mu + i*omega) z + c |z|^2 z - |z|^4 z] dt + sigma dW,   dW = dW1 + i dW2
The rotational drift is divergence-free and orthogonal to grad V, so the stationary
density is p(z) ~ exp(-2 V(|z|)/sigma^2), V(r) = -(mu r^2/2 + c r^4/4 - r^6/6).
With P = |z|^2 (uniform area element in P):
    p(P) ~ exp(a1 P + a2 P^2 + a3 P^3),  a1 = mu/s2, a2 = c/(2 s2), a3 = -1/(3 s2),  s2 = sigma^2
i.e. a three-parameter exponential family on P >= 0 (convex MLE).  Amplitude units are
arbitrary, so only the two noise-scaled unfolding parameters are identifiable:
    mu_t = mu * sigma^(-4/3),   c_t = c * sigma^(-2/3)
recovered as s = (-1/(3 a3))^(1/3), mu_t = a1 s, c_t = 2 a2 s^2.
Regimes (modes of p(P) coincide with deterministic attractors):
    mu_t > 0                               -> oscillatory (limit cycle)
    mu_t < 0, c_t > 0, c_t^2 + 4 mu_t > 0  -> bistable (focus + cycle)
    otherwise                              -> noisy focus (damped / linear-like)
"""
import numpy as np
from scipy.optimize import minimize

GRID_N = 6000


def _grid(q):
    qmax = max(60.0, 3.0 * float(np.max(q)))
    g = np.linspace(0.0, qmax, GRID_N)
    w = np.full(GRID_N, g[1] - g[0]); w[0] *= 0.5; w[-1] *= 0.5
    return g, w


def _nll_and_grad(b, tbar, G, w):
    e = b[0] * G[0] + b[1] * G[1] + b[2] * G[2]
    m = e.max()
    pe = np.exp(e - m) * w
    Z = pe.sum()
    logZ = m + np.log(Z)
    mom = np.array([(pe * G[k]).sum() / Z for k in range(3)])
    return logZ - b @ tbar, mom - tbar


def fit_power(P, b3_max=-1e-5):
    """MLE of natural parameters (a1,a2,a3) on mean-normalised power q = P/mean(P)."""
    P = np.asarray(P, float)
    P = P[np.isfinite(P) & (P >= 0)]
    q = P / P.mean()
    tbar = np.array([q.mean(), (q ** 2).mean(), (q ** 3).mean()])
    g, w = _grid(q)
    G = np.stack([g, g ** 2, g ** 3])
    x0 = np.array([-1.0, 0.0, -1e-3])
    res = minimize(lambda b: _nll_and_grad(b, tbar, G, w), x0, jac=True, method="L-BFGS-B",
                   bounds=[(-200, 200), (-200, 200), (-200, b3_max)],
                   options=dict(maxiter=2000, ftol=1e-13, gtol=1e-9))
    a1, a2, a3 = res.x
    return dict(a1=a1, a2=a2, a3=a3, nll=res.fun, ok=bool(res.success), **to_bautin(a1, a2, a3))


def fit_power_deconv(Q, rho, nbins=300, b3_max=-1e-5, x0=None):
    """MLE of the oscillator's (a1,a2,a3) when the observed analytic power Q = |z + eta|^2 contains
    independent circular-Gaussian background eta with E|eta|^2 = rho * E[Q]  (0 <= rho < 1).
    p(Q) = int p_z(P) (1/s) exp(-(Q+P)/s) I0(2 sqrt(QP)/s) dP,  s = rho (Q normalised to mean 1).
    Binned (multinomial) likelihood on a fine histogram of Q; latent P integrated on a grid."""
    from scipy.special import i0e
    Q = np.asarray(Q, float); Q = Q[np.isfinite(Q)]
    q = Q / Q.mean()
    if rho < 1e-3:
        return fit_power(q, b3_max)
    s = float(rho)
    edges = np.quantile(q, np.linspace(0, 1, nbins + 1)); edges[0] = 0.0; edges[-1] = q.max() * 1.0001
    edges = np.unique(edges)
    cnt, _ = np.histogram(q, edges)
    wts = cnt / cnt.sum()
    qc = 0.5 * (edges[1:] + edges[:-1]); dq = np.diff(edges)
    pmax = max(40.0, 3 * q.max())
    pg = np.linspace(0, pmax, 800); dp = pg[1] - pg[0]
    # Rician kernel K[i,j] = p(Q=qc_i | P=pg_j) = exp(-(sqrt q - sqrt p)^2/s) i0e(2 sqrt(qp)/s) / s  (stable)
    K = np.exp(-(np.sqrt(qc)[:, None] - np.sqrt(pg)[None, :]) ** 2 / s) * i0e(2.0 * np.sqrt(np.outer(qc, pg)) / s) / s
    K *= dp
    G = np.stack([pg, pg ** 2, pg ** 3])

    def nll(b):
        e = b @ G
        pz = np.exp(e - e.max()); pz /= pz.sum() * dp
        W = K * pz[None, :]
        dens = W.sum(1) + 1e-300
        f = -(wts * np.log(dens * dq + 1e-300)).sum()
        Ez = (G * pz).sum(1) * dp
        cond = (W @ G.T) / dens[:, None]           # E[G_k | bin i]
        g = -(wts[:, None] * (cond - Ez[None, :])).sum(0)
        return f, g

    starts = [x0] if x0 is not None else []
    starts += [np.array([-1.0 / (1 - s), 0.0, -1e-3]), np.array([0.5, 0.5, -0.3]), np.array([-1.0, 1.0, -0.5])]
    best = None
    for st in starts:
        r = minimize(nll, st, jac=True, method="L-BFGS-B", bounds=[(-60, 60), (-60, 60), (-60, b3_max)],
                     options=dict(maxiter=1000))
        if best is None or r.fun < best.fun:
            best = r
    a1, a2, a3 = best.x
    return dict(a1=a1, a2=a2, a3=a3, nll=best.fun, ok=bool(best.success), rho=s, **to_bautin(a1, a2, a3))


def to_bautin(a1, a2, a3):
    s = (-1.0 / (3.0 * a3)) ** (1.0 / 3.0)
    mu_t = a1 * s
    c_t = 2.0 * a2 * s * s
    return dict(mu_t=mu_t, c_t=c_t, regime=regime(mu_t, c_t))


def regime(mu_t, c_t):
    if mu_t > 0:
        return "oscillatory"
    if c_t > 0 and c_t * c_t + 4 * mu_t > 0:
        return "bistable"
    return "focus"


def fold_delta(mu_t, c_t):
    """Signed distance to the fold of limit cycles, mu + c^2/4 for c > 0 (negative on the focus side);
    for c <= 0 the fold does not exist and the distance to the Hopf line, mu, is returned."""
    mu_t, c_t = np.asarray(mu_t, float), np.asarray(c_t, float)
    return np.where(c_t > 0, mu_t + c_t * c_t / 4, mu_t)


def fit_linear_null(P):
    """Exponential (Rayleigh-amplitude) null: a2=a3=0. Returns nll on normalised q."""
    q = np.asarray(P, float); q = q / q.mean()
    return float(np.mean(q))  # -loglik of Exp(1) per sample = mean(q) (log rate = 0)


def envelope_cv(P):
    a = np.sqrt(np.asarray(P, float))
    return float(a.std() / a.mean())


from numba import njit


@njit(cache=True)
def _em_bautin(mu, c, sigma, om, dt, n, step, seed, mult):
    np.random.seed(seed)
    zr, zi = 0.1, 0.0
    m = n // step
    outr = np.empty(m); outi = np.empty(m)
    sq = sigma * np.sqrt(dt)
    k = 0
    for i in range(n):
        r2 = zr * zr + zi * zi
        g = mu + c * r2 - r2 * r2
        # optional multiplicative component: noise amplitude scales with (1 + mult*|z|)
        amp = sq * (1.0 + mult * np.sqrt(r2))
        dzr = (g * zr - om * zi) * dt + amp * np.random.randn()
        dzi = (g * zi + om * zr) * dt + amp * np.random.randn()
        zr += dzr; zi += dzi
        if (i + 1) % step == 0 and k < m:
            outr[k] = zr; outi[k] = zi; k += 1
    return outr[:k] + 1j * outi[:k]


@njit(cache=True)
def _em_bautin_mod(mus, sigmas, c, om, dt, step, seed):
    """Same SDE with piecewise-constant (per output sample) mu(t) and sigma(t)."""
    np.random.seed(seed)
    zr, zi = 0.1, 0.0
    m = mus.size
    outr = np.empty(m); outi = np.empty(m)
    for k in range(m):
        mu = mus[k]; sq = sigmas[k] * np.sqrt(dt)
        for j in range(step):
            r2 = zr * zr + zi * zi
            g = mu + c * r2 - r2 * r2
            dzr = (g * zr - om * zi) * dt + sq * np.random.randn()
            dzi = (g * zi + om * zr) * dt + sq * np.random.randn()
            zr += dzr; zi += dzi
        outr[k] = zr; outi[k] = zi
    return outr + 1j * outi


@njit(cache=True)
def _em_rot(mus, sigmas, c, dt, step, seed):
    """Rotating-frame integrator: w = z exp(-i omega t) obeys dw = (mu + c|w|^2 - |w|^4) w dt + sigma dW
    (isotropic complex noise is rotation invariant in law), so the carrier imposes no step-size limit.
    Caller multiplies the output by exp(i omega t_k)."""
    np.random.seed(seed)
    wr, wi = 0.1, 0.0
    m = mus.size
    outr = np.empty(m); outi = np.empty(m)
    for k in range(m):
        mu = mus[k]; sq = sigmas[k] * np.sqrt(dt)
        for j in range(step):
            r2 = wr * wr + wi * wi
            g = mu + c * r2 - r2 * r2
            wr += g * wr * dt + sq * np.random.randn()
            wi += g * wi * dt + sq * np.random.randn()
        outr[k] = wr; outi[k] = wi
    return outr + 1j * outi


@njit(cache=True)
def _em_rot_mult(mus, sigmas, c, kmult, dt, step, seed):
    """Rotating-frame integrator with additive plus state-dependent (multiplicative) noise, as in the canonical
    multistability model of Freyer et al. (2012): sigma dW1 + sigma * kmult * w dW2 with independent isotropic
    complex increments, i.e. isotropic noise of amplitude sigma * sqrt(1 + kmult^2 |w|^2) (Ito).
    kmult is given in physical units of w; callers pass k * sigma^(-1/3) for a scale-free ratio k."""
    np.random.seed(seed)
    wr, wi = 0.1, 0.0
    m = mus.size
    outr = np.empty(m); outi = np.empty(m)
    for k in range(m):
        mu = mus[k]; sq = sigmas[k] * np.sqrt(dt)
        for j in range(step):
            r2 = wr * wr + wi * wi
            g = mu + c * r2 - r2 * r2
            amp = sq * np.sqrt(1.0 + kmult * kmult * r2)
            wr += g * wr * dt + amp * np.random.randn()
            wi += g * wi * dt + amp * np.random.randn()
        outr[k] = wr; outi[k] = wi
    return outr + 1j * outi


def radial_rate_max(mu_t, c_t, t0):
    """max |d/dr radial drift| = |mu_t + 3 c_t u - 5 u^2| / t0 over the stationary support of u (1/s)."""
    u = np.linspace(0, 60, 6001)
    e = mu_t * u + c_t * u ** 2 / 2 - u ** 3 / 3
    sup = u[e > e.max() - 30]
    umax = sup.max() * 1.2 + 0.5
    uu = np.linspace(0, umax, 400)
    return float(np.max(np.abs(mu_t + 3 * c_t * uu - 5 * uu * uu)) / t0)


def ou_series(n, fs, tau_s, rng):
    """Unit-variance Ornstein-Uhlenbeck sample path with correlation time tau_s."""
    a = np.exp(-1.0 / (fs * tau_s)); x = np.empty(n); x[0] = rng.standard_normal()
    e = rng.standard_normal(n) * np.sqrt(1 - a * a)
    for i in range(1, n):
        x[i] = a * x[i - 1] + e[i]
    return x


def simulate_modulated(mu_t0, c_t, sigma, mod_sd, mod_tau, f0=10.0, fs=250.0, T=120.0, dt=2e-4, seed=0, burn=5.0,
                       target="mu"):
    """Bautin SDE whose unfolding parameter mu_t (or noise intensity) is slowly modulated by an OU
    'vigilance' process: mu_t(t) = mu_t0 + mod_sd * OU(t)  (target='mu'), or
    sigma(t) = sigma * exp(mod_sd * OU(t)) (target='sigma'). Hidden slow variable -> not 1-D Markov."""
    rng = np.random.default_rng(seed + 12345)
    n = int((T + burn) * fs)
    o = ou_series(n, fs, mod_tau, rng)
    if target == "mu":
        mus = (mu_t0 + mod_sd * o) * sigma ** (4 / 3); sig = np.full(n, sigma)
    else:
        mus = np.full(n, mu_t0 * sigma ** (4 / 3)); sig = sigma * np.exp(mod_sd * o)
    step = int(round(1.0 / (fs * dt)))
    z = _em_bautin_mod(mus, sig, float(c_t * sigma ** (2 / 3)), 2 * np.pi * f0, dt, step, int(seed))
    return z[int(burn * fs):]


def simulate_bautin(mu, c, sigma, f0=10.0, fs=250.0, T=120.0, dt=2e-4, seed=0, burn=5.0, mult=0.0):
    """Euler-Maruyama of the complex Bautin SDE; returns complex z sampled at fs (x = Re z)."""
    n = int((T + burn) / dt)
    step = int(round(1.0 / (fs * dt)))
    z = _em_bautin(float(mu), float(c), float(sigma), 2 * np.pi * f0, dt, n, step, int(seed), float(mult))
    return z[int(burn * fs):]

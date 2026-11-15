"""TDBRAIN V3.1 loading + light, deterministic preprocessing."""
import os, glob, numpy as np, pandas as pd, mne
from .paths import TDBRAIN_ROOT as ROOT, TDBRAIN_META as META
EEG26 = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC3", "FCz", "FC4", "T7", "C3", "Cz", "C4", "T8",
         "CP3", "CPz", "CP4", "P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2"]
FS = 250.0


def load_meta():
    df = pd.read_excel(META)
    df = df.rename(columns={"TDBRAIN_ID": "sub"})
    return df


def rest_file(sub, task="restEC", ses=1):
    f = f"{ROOT}/{sub}/ses-{ses}/eeg/{sub}_ses-{ses}_task-{task}_eeg.bdf"
    return f if os.path.exists(f) else None


EOG = ["VPVA", "VNVB", "HPHL", "HNHR"]


def load_rest(path, l_freq=0.5, h_freq=45.0, thr_uv=150.0, pad_s=0.5, edge_s=3.0, variant=None):
    """Returns X (26 x n, microvolts, 250 Hz, linked-mastoid reference as recorded),
    per-channel good-sample mask M (26 x n) and a bad-channel flag vector.
    Bad channel: robust z of log-std across channels > 3.5, or flat, or >30% samples over threshold.
    Bad samples: |x| > thr_uv on that channel, padded by pad_s; first/last edge_s seconds dropped.
    variant='eog': vertical and horizontal EOG (bipolar) regressed out of every channel before masking.
    variant='csd': surface Laplacian of the good channels (masks: all contributing channels clean)."""
    raw = mne.io.read_raw_bdf(path, preload=True, verbose="ERROR")
    picks = [c for c in EEG26 if c in raw.ch_names]
    eog = [c for c in EOG if c in raw.ch_names] if variant == "eog" else []
    raw.pick(picks + eog)
    raw.filter(l_freq, h_freq, verbose="ERROR")
    raw.resample(FS, verbose="ERROR")
    X = raw.get_data() * 1e6
    if variant == "eog":
        if len(eog) < 4:
            raise ValueError("EOG channels missing")
        E = np.vstack([X[len(picks)] - X[len(picks) + 1], X[len(picks) + 2] - X[len(picks) + 3]])
        X = X[:len(picks)]
        e0 = int(edge_s * FS); seg = slice(e0, X.shape[1] - e0)
        B = np.linalg.lstsq(E[:, seg].T, X[:, seg].T, rcond=None)[0]   # 2 x channels propagation coefficients
        X = X - B.T @ E
    n = X.shape[1]
    M = np.abs(X) <= thr_uv
    k = int(pad_s * FS)
    ker = np.ones(2 * k + 1)
    for i in range(X.shape[0]):
        if not M[i].all():
            M[i] = np.convolve((~M[i]).astype(float), ker, "same") == 0
    e = int(edge_s * FS); M[:, :e] = False; M[:, n - e:] = False
    ls = np.log(X[:, e:n - e].std(1) + 1e-9)
    med = np.median(ls); mad = np.median(np.abs(ls - med)) * 1.4826 + 1e-9
    badch = (np.abs(ls - med) / mad > 3.5) | (np.exp(ls) < 0.5) | (M[:, e:n - e].mean(1) < 0.7)
    if variant == "csd":
        from .csd import operator, support
        g = np.where(~badch)[0]
        W = operator([picks[i] for i in g])
        S = support(W)
        X = X.copy(); X[g] = W @ X[g]
        Mg = M[g].copy()
        for r, i in enumerate(g):
            M[i] = Mg[S[r]].all(0)
    return X, M, badch, picks

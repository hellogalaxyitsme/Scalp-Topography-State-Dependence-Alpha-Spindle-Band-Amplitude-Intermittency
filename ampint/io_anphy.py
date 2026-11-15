"""ANPHY-Sleep (Wei et al. 2024): healthy nights of 83-channel HD-EEG at 1 kHz with 30-s sleep staging.
Each night is extracted to a temporary directory outside the dataset, converted once to a 250-Hz
average-referenced float32 array (cache), and the temporary extract is removed."""
import os, glob, json, shutil, subprocess, numpy as np, mne
from .paths import ANPHY_ROOT as ROOT, ANPHY_TMP as TMP, ANPHY_CACHE as CACHE
FS = 250.0
NON_EEG = {"SO1", "SO2", "ZY1", "ZY2", "CHEMG1", "CHEMG2", "RLEG", "RLEG+", "LLEG", "LLEG+",
           "EOG1", "EOG2", "ECG1", "ECG2"}  # compared after norm(); 'RLEG-' normalises to 'RLEG'
STAGE_MAP = {"W": "W", "N1": "N1", "N2": "N2", "N3": "N3", "R": "REM"}
EPOCH = 30.0


def norm(name):
    """Channel names vary across nights ('C3', 'C3-REF', 'F4-'); return the plain upper-case label."""
    n = name.upper()
    return n[:-4] if n.endswith("-REF") else n.rstrip("-")


def subjects():
    return sorted(os.path.basename(z)[:7] for z in glob.glob(f"{ROOT}/EPCTL*.zip"))


def positions():
    """Average co-registered electrode positions (name -> xyz in m)."""
    pos = {}
    for line in open(f"{ROOT}/Co-registered average positions.pos"):
        t = line.split()
        if len(t) >= 5:
            pos[t[1].upper()] = np.array([float(v) for v in t[2:5]]) / 100.0
    return pos


def head_positions():
    """Positions in MNE head coordinates (x right, y nasion, z up; m), centred on a least-squares sphere,
    and the sphere radius. The co-registered file uses x anterior, y left."""
    pos = positions()
    names = list(pos); xyz = np.array([pos[n] for n in names])
    G = np.column_stack([2 * xyz, np.ones(len(xyz))])
    sol = np.linalg.lstsq(G, np.sum(xyz ** 2, 1), rcond=None)[0]
    c0 = sol[:3]; r0 = float(np.sqrt(sol[3] + c0 @ c0))
    return {n: np.array([-(p[1] - c0[1]), p[0] - c0[0], p[2] - c0[2]]) for n, p in zip(names, xyz)}, r0


def _epochs(txt):
    """30-s stage labels from the scoring file (tokens: label, onset s, duration s)."""
    tok = open(txt).read().split()
    lab = {}
    for i in range(0, len(tok) - 2, 3):
        st, on, du = tok[i], float(tok[i + 1]), float(tok[i + 2])
        for k in range(int(round(du / EPOCH))):
            lab[int(round(on / EPOCH)) + k] = STAGE_MAP.get(st, "?")
    n = max(lab) + 1
    return np.array([lab.get(i, "?") for i in range(n)], dtype=object)


def prepare(sid, batch=10, thr_uv=200.0):
    """Extract, filter (0.5-45 Hz), resample to 250 Hz, flag bad channels, average-reference; cache to disk."""
    out = f"{CACHE}/{sid}"
    if os.path.exists(f"{out}_meta.json"):
        return
    os.makedirs(CACHE, exist_ok=True)
    zp = glob.glob(f"{ROOT}/{sid}*.zip")[0]
    tmp = f"{TMP}/{sid}"
    os.makedirs(tmp, exist_ok=True)
    subprocess.run(["unzip", "-o", "-q", zp, "-d", tmp], check=True)
    try:
        edf = glob.glob(f"{tmp}/**/*.edf", recursive=True)[0]
        txt = glob.glob(f"{tmp}/**/*.txt", recursive=True)[0]
        raw = mne.io.read_raw_edf(edf, preload=False, verbose="ERROR")
        chans = [c for c in raw.ch_names if norm(c) not in NON_EEG and c != "EDF Annotations"]
        parts = []
        for i in range(0, len(chans), batch):
            r = raw.copy().pick(chans[i:i + batch]).load_data(verbose="ERROR")
            r.filter(0.5, 45.0, verbose="ERROR")
            r.resample(FS, method="polyphase", verbose="ERROR")
            parts.append((r.get_data() * 1e6).astype(np.float32))
            del r
        X = np.concatenate(parts, 0)
        epochs = _epochs(txt)
        # bad channels: flat, or robust z of log-std (over in-bed scored epochs) > 3.5
        scored = np.repeat(np.isin(epochs, list(STAGE_MAP.values())), int(EPOCH * FS))[:X.shape[1]]
        sd = np.log(np.std(X[:, np.where(scored)[0][::10]], axis=1) + 1e-9)
        z = (sd - np.median(sd)) / (1.4826 * np.median(np.abs(sd - np.median(sd))) + 1e-12)
        bad = (np.abs(z) > 3.5) | (np.exp(sd) < 0.5)
        X -= X[~bad].mean(0, keepdims=True)
        np.save(f"{out}_X.npy", X)
        json.dump(dict(ch=[norm(c) for c in chans], bad=bad.tolist(), epochs=epochs.tolist(), fs=FS,
                       n=int(X.shape[1])), open(f"{out}_meta.json", "w"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def load(sid, wake_margin_min=30.0, thr_uv=200.0, pad_s=0.5):
    """Memory-mapped X (channels x n, uV, 250 Hz, average reference), per-sample stage labels, channel names,
    bad-channel flags, and a function returning the artefact mask of one channel.
    Wake is restricted to within wake_margin_min of the first/last sleep epoch, as for Sleep-EDF."""
    m = json.load(open(f"{CACHE}/{sid}_meta.json"))
    X = np.load(f"{CACHE}/{sid}_X.npy", mmap_mode="r")
    ep = np.array(m["epochs"], dtype=object)
    sl = np.where(np.isin(ep, ["N1", "N2", "N3", "REM"]))[0]
    k = int(wake_margin_min * 60 / EPOCH)
    ep[:max(0, sl[0] - k)] = "?"; ep[sl[-1] + k + 1:] = "?"
    lab = np.repeat(ep, int(EPOCH * FS))
    n = X.shape[1]
    lab = np.r_[lab, np.full(max(0, n - lab.size), "?", dtype=object)][:n]

    def mask(ci):
        M = np.abs(X[ci]) <= thr_uv
        kk = int(pad_s * FS)
        return np.convolve((~M).astype(float), np.ones(2 * kk + 1), "same") == 0
    return X, lab, [norm(c) for c in m["ch"]], np.array(m["bad"]), mask

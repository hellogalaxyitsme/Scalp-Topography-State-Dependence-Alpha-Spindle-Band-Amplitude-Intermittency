"""Sleep-EDF Expanded (sleep-cassette) loading: EEG Fpz-Cz / Pz-Oz at 100 Hz + 30-s hypnogram."""
import os, glob, numpy as np, mne
from .paths import SLEEPEDF_ROOT as ROOT
FS = 250.0  # recordings are 100 Hz; upsampled to 250 Hz so the same simulation banks/pipeline apply
STAGE_MAP = {"Sleep stage W": "W", "Sleep stage 1": "N1", "Sleep stage 2": "N2", "Sleep stage 3": "N3",
             "Sleep stage 4": "N3", "Sleep stage R": "REM"}
CHANS = ["EEG Fpz-Cz", "EEG Pz-Oz"]


def pairs():
    out = []
    for psg in sorted(glob.glob(f"{ROOT}/SC*-PSG.edf")):
        stem = os.path.basename(psg)[:7]  # SC4ssN
        hyp = glob.glob(f"{ROOT}/{stem}*-Hypnogram.edf")
        if hyp:
            out.append((psg, hyp[0]))
    return out


def load(psg, hyp, wake_margin_min=30.0, thr_uv=200.0, pad_s=0.5):
    """Returns X (2 x n, uV, 100 Hz, 0.5-45 Hz), per-sample stage labels (str array), artefact mask (2 x n).
    Wake is restricted to within wake_margin_min of the first/last sleep epoch (standard trimming)."""
    raw = mne.io.read_raw_edf(psg, preload=True, verbose="ERROR", include=CHANS)
    raw.filter(0.5, 45.0, verbose="ERROR")
    raw.resample(FS, verbose="ERROR")
    X = raw.get_data(picks=CHANS) * 1e6
    n = X.shape[1]
    ann = mne.read_annotations(hyp)
    lab = np.full(n, "?", dtype=object)
    for on, du, de in zip(ann.onset, ann.duration, ann.description):
        if de in STAGE_MAP:
            a = int(round(on * FS)); b = min(n, int(round((on + du) * FS)))
            lab[a:b] = STAGE_MAP[de]
    sl = np.where(np.isin(lab, ["N1", "N2", "N3", "REM"]))[0]
    if sl.size:
        m = int(wake_margin_min * 60 * FS)
        lab[:max(0, sl[0] - m)] = "?"; lab[sl[-1] + m:] = "?"
    M = np.abs(X) <= thr_uv
    k = int(pad_s * FS); ker = np.ones(2 * k + 1)
    for i in range(2):
        M[i] = np.convolve((~M[i]).astype(float), ker, "same") == 0
    return X, lab, M

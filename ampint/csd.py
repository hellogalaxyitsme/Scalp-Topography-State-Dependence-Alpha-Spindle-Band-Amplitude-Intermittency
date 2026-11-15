"""Surface Laplacian (current source density; spherical splines, Perrin et al. 1989, as implemented in MNE)
expressed as a linear operator on the good channels, so it can be applied to any segment: csd = W @ x."""
import numpy as np, mne


def operator(names, ch_pos=None):
    """names: channel names; ch_pos: dict name -> xyz in head coordinates (m), or None for standard 10-20 positions."""
    info = mne.create_info(list(names), 250.0, "eeg")
    if ch_pos is None:
        info.set_montage("standard_1020", match_case=False)
    else:
        info.set_montage(mne.channels.make_dig_montage(ch_pos={n: ch_pos[n] for n in names}, coord_frame="head"))
    raw = mne.io.RawArray(np.eye(len(names)), info, verbose="ERROR")
    return mne.preprocessing.compute_current_source_density(raw, verbose="ERROR").get_data()


def support(W, frac=0.1):
    """Channels that contribute to each Laplacian channel (|weight| >= frac x the largest weight of its row)."""
    A = np.abs(W)
    return A >= frac * A.max(1, keepdims=True)

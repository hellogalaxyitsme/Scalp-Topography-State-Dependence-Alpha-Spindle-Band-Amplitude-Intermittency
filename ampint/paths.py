"""Locations of the raw datasets and of the ANPHY-Sleep working directories. Each path can be set with the
environment variable of the same name; the defaults are relative to the repository root."""
import os


def _p(name, default):
    return os.environ.get(name, default)


TDBRAIN_ROOT = _p("TDBRAIN_ROOT", "data/TDBRAIN_Dataset_V3_1")                 # BIDS tree: sub-*/ses-*/eeg/*.bdf
TDBRAIN_META = _p("TDBRAIN_META", "data/TDBRAIN_participants_V3.xlsx")         # participants table (V3)
SLEEPEDF_ROOT = _p("SLEEPEDF_ROOT", "data/sleep-edfx-1.0.0/sleep-cassette")    # PSG and hypnogram EDF files
ANPHY_ROOT = _p("ANPHY_ROOT", "data/ANPHY_Sleep")                              # EPCTL*.zip, electrode positions
ANPHY_TMP = _p("ANPHY_TMP", "tmp/anphy")                                       # temporary extraction directory
ANPHY_CACHE = _p("ANPHY_CACHE", "cache_anphy")                                 # 250 Hz average-reference cache
MUMTAZ_ROOT = _p("MUMTAZ_ROOT", "data/Mumtaz")                                 # EDF files of the figshare release
DS003478_ROOT = _p("DS003478_ROOT", "data/ds003478")                           # OpenNeuro ds003478 (BIDS)

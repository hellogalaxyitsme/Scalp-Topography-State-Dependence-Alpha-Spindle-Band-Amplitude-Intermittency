"""Parameter-matched coverage percentile for the ANPHY-Sleep posterior tables (as coverage_rows.py).
usage: python scripts/coverage_anphy.py [variant]   (variant: '' or 'csd')"""
import sys, pandas as pd
from coverage_rows import scorer
from ampint.summ import FEATURES
from ood_matched import PAR

SFX = f"_{sys.argv[1]}" if len(sys.argv) > 1 and sys.argv[1] else ""
for band, cfg in (("alpha", "alpha"), ("sigma", "sigma250")):
    e = pd.read_csv(f"results/post_anphy{SFX}_{band}.csv", usecols=FEATURES + PAR)
    pd.DataFrame(dict(cov_pct=scorer(cfg)(e))).to_csv(f"results/coverage_anphy{SFX}_{band}.csv", index=False)
    print(band, len(e), flush=True)

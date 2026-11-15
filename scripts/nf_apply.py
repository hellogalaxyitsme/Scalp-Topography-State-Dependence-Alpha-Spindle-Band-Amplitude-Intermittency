"""Apply a trained amortised posterior to a table of summary statistics.
usage: python scripts/nf_apply.py <model.joblib> <input_dir_or_csv> <out.csv> [col=value ...]
(an input directory means all *.csv in it; the output may not be written inside the input directory)"""
import sys, os, glob, joblib, numpy as np, pandas as pd
from sbi_fit import predict

model, src, out = sys.argv[1], sys.argv[2], sys.argv[3]
filters = sys.argv[4:]
assert out.endswith(".csv") and all("=" in f for f in filters), "bad arguments"
if os.path.isdir(src):
    assert os.path.abspath(os.path.dirname(out)) != os.path.abspath(src), "refusing to write into the input directory"
    files = sorted(glob.glob(os.path.join(src, "*.csv")))
else:
    assert os.path.abspath(out) != os.path.abspath(src)
    files = [src]
M = joblib.load(model)
df = pd.concat([pd.read_csv(f) for f in files if open(f).readline().strip()], ignore_index=True)
for flt in filters:
    k, v = flt.split("=", 1)
    df = df[df[k].astype(str) == v]
df = df.replace([np.inf, -np.inf], np.nan)
df = df[df[M["features"]].notna().all(1)].reset_index(drop=True)
pr = predict(M, df)
pd.concat([df, pr.reset_index(drop=True)], axis=1).to_csv(out, index=False)
print(len(df), "rows ->", out)

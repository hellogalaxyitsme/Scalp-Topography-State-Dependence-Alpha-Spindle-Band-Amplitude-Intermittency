"""Adequacy-test power and parameter bias on simulations with a hidden slow (vigilance-like) modulation."""
import sys, json, joblib, numpy as np, pandas as pd
from sbi_fit import load_bank, predict

cfg = sys.argv[1] if len(sys.argv) > 1 else "alpha"
M = joblib.load(f"results/sbi_{cfg}_full.joblib")
mb = load_bank(cfg, "modbank").reset_index(drop=True)
pm = predict(M, mb)
d = pd.concat([mb, pm], axis=1)
out = dict(n=int(len(d)), flag_rate=float(d.slow_flag.mean()))
out["flag_rate_by_sd"] = {str(k): float(v) for k, v in d.groupby(pd.cut(d.mod_sd, [0, .5, 1, 1.5, 2.0, 2.6])).slow_flag.mean().items()}
out["flag_rate_by_tau"] = {str(k): float(v) for k, v in d.groupby(pd.cut(d.mod_tau, [0, 5, 10, 20, 31])).slow_flag.mean().items()}
out["bias_c_median"] = float(np.median(d.th_c_q50 - d.th_c)); out["bias_mu_median"] = float(np.median(d.th_mu_q50 - d.th_mu))
out["bias_c_flagged"] = float(np.median((d.th_c_q50 - d.th_c)[d.slow_flag == 1]))
out["bias_c_unflagged"] = float(np.median((d.th_c_q50 - d.th_c)[d.slow_flag == 0]))
# unmodulated held-out false-flag rate (from training report)
rep = json.load(open(f"results/sbi_{cfg}_full_report.json"))
out["false_flag_rate_unmodulated"] = rep.get("tail_false_flag_rate")
d.to_csv(f"results/sbi_{cfg}_modulated_eval.csv", index=False)
json.dump(out, open(f"results/sbi_{cfg}_modulated_eval.json", "w"), indent=1)
print(json.dumps(out, indent=1))

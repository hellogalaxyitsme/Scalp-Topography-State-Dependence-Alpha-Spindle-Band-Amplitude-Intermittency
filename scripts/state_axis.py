"""State axis in the unfolding plane. The axis is the within-participant eyes-open displacement of resting alpha
(TDBRAIN healthy, fixed before the clinical contrasts were projected). It is checked against an independent
within-participant arousal change, wake -> N1 drowsiness of the alpha rhythm (Sleep-EDF; ANPHY-Sleep when
available). Every contrast is decomposed into a component along the axis and an orthogonal component, with
bootstrap intervals that include the uncertainty of the axis (paired bootstrap replicates).
Decision rule: a clinical contrast is called disorder-specific only if its orthogonal component is significant
(bootstrap p, FDR over the clinical contrasts)."""
import os, json, numpy as np, pandas as pd
from displacements import contrasts, boot_mean, cos, EO, NB
from ampint.stats import fdr

ANPHY_POST = ["O1", "OZ", "O2", "POZ", "PZ", "PO3", "PO4"]


def paired(df, a, b, idx):
    w = df.pivot_table(index=idx, columns="stage", values=["th_mu_q50", "th_c_q50"])
    return boot_mean(np.column_stack([w[("th_mu_q50", b)] - w[("th_mu_q50", a)], w[("th_c_q50", b)] - w[("th_c_q50", a)]]))


def pboot(b):
    return float(min(1.0, max(1.0 / len(b), 2 * min(np.mean(b <= 0), np.mean(b >= 0)))))


def main():
    out, BOOT = contrasts()
    sa = pd.read_csv("results/sleep_alpha_subject_stage.csv")
    for ch, tag in (("EEG Pz-Oz", "Pz-Oz"), ("EEG Fpz-Cz", "Fpz-Cz")):
        r = paired(sa[sa.ch == ch], "W", "N1", "subj")
        out[f"alpha wake -> N1 (Sleep-EDF {tag})"], BOOT[f"alpha wake -> N1 (Sleep-EDF {tag})"] = r
    if os.path.exists("results/anphy_alpha_subject_stage.csv"):
        aa = pd.read_csv("results/anphy_alpha_subject_stage.csv")
        aa = aa[aa.ch.isin(ANPHY_POST)].groupby(["rec", "stage"])[["th_mu_q50", "th_c_q50"]].mean().reset_index()
        r = paired(aa, "W", "N1", "rec")
        out["alpha wake -> N1 (ANPHY posterior)"], BOOT["alpha wake -> N1 (ANPHY posterior)"] = r
    beo = BOOT[EO]
    u = np.array([out[EO]["d_mu"], out[EO]["d_c"]]); u /= np.linalg.norm(u)
    ub = beo / np.linalg.norm(beo, axis=1, keepdims=True)
    perp = lambda v: np.stack([-v[..., 1], v[..., 0]], -1)
    res = dict(axis=dict(u_mu=float(u[0]), u_c=float(u[1]), angle_deg=float(np.degrees(np.arctan2(u[1], u[0])))))
    for k, v in out.items():
        d = np.array([v["d_mu"], v["d_c"]]); b = BOOT[k]
        al, orth = d @ u, d @ perp(u)
        alb, orb = np.sum(b * ub, 1), np.sum(b * perp(ub), 1)
        res[k] = dict(n=v["n"], d_mu=v["d_mu"], d_c=v["d_c"], along=float(al), along_lo=float(np.percentile(alb, 2.5)),
                      along_hi=float(np.percentile(alb, 97.5)), p_along=pboot(alb), orth=float(orth),
                      orth_lo=float(np.percentile(orb, 2.5)), orth_hi=float(np.percentile(orb, 97.5)), p_orth=pboot(orb),
                      cos_eo=float(cos(d, u)), cos_lo=float(np.percentile(cos(b, beo), 2.5)),
                      cos_hi=float(np.percentile(cos(b, beo), 97.5)))
    clin = [k for k in res if k.startswith(("TDBRAIN", "Mumtaz", "ds003478"))]
    for kind in ("along", "orth"):
        q = fdr([res[k][f"p_{kind}"] for k in clin])
        for k, qq in zip(clin, q):
            res[k][f"q_{kind}"] = float(qq)
    json.dump(res, open("results/state_axis.json", "w"), indent=1)
    print(f"axis (EO): u = ({u[0]:+.3f}, {u[1]:+.3f}), angle {res['axis']['angle_deg']:.1f} deg")
    for k, r in res.items():
        if k == "axis":
            continue
        print(f"{k:40s} n={r['n']!s:>8}  along={r['along']:+.2f} [{r['along_lo']:+.2f},{r['along_hi']:+.2f}] p={r['p_along']:.4f}"
              f"{' q=%.3f' % r['q_along'] if 'q_along' in r else '':8s}  orth={r['orth']:+.2f} [{r['orth_lo']:+.2f},{r['orth_hi']:+.2f}] "
              f"p={r['p_orth']:.4f}{' q=%.3f' % r['q_orth'] if 'q_orth' in r else ''}  cos={r['cos_eo']:+.3f} [{r['cos_lo']:+.2f},{r['cos_hi']:+.2f}]")


if __name__ == "__main__":
    main()

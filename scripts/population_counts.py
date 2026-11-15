"""Analysis populations: participants, recordings and analysis units entering each analysis.
usage: python scripts/population_counts.py  -> results/population_counts.json"""
import json, numpy as np, pandas as pd
from ampint.io_tdbrain import load_meta


def stage_counts(P, unit):
    return {s: int(n) for s, n in P.groupby("stage")[unit].nunique().items()}


def main():
    out = {}
    meta = load_meta()
    m1 = meta[meta.sessID == 1].drop_duplicates("sub")[["sub", "indication", "age", "gender"]]
    P = pd.read_csv("results/post_tdbrain.csv", usecols=["sub", "ses", "task", "ch", "kind"])
    d1 = P[(P.kind == "data") & (P.ses == "ses-1")].merge(m1, on="sub", how="left")
    adult = d1[(d1.age >= 18) & d1.gender.notna()]
    R = pd.read_csv("results/nf_tdbrain_reclevel.csv")
    t = {}
    for task in ("restEC", "restEO"):
        H = adult[(adult.indication == "HEALTHY") & (adult.task == task)]
        t[f"healthy {task}"] = dict(participants=int(H["sub"].nunique()), channel_recordings=int(len(H)))
    hp = R[(R.group == "HEALTHY")].pivot_table(index="sub", columns="task", values="th_mu_q50").dropna()
    t["healthy, both conditions"] = int(len(hp))
    t["clinical groups (EC)"] = {g: int(n) for g, n in R[(R.task == "restEC") & (R.group != "HEALTHY")].group.value_counts().items()}
    tw = P[(P.kind == "iaaft0") & (P.ses == "ses-1")].merge(m1, on="sub", how="left")
    tw = tw[(tw.indication == "HEALTHY") & (tw.age >= 18) & tw.gender.notna()]
    t["healthy twins (O1, Oz, O2, Pz), channel recordings per twin"] = {k: int(n) for k, n in tw.task.value_counts().items()}
    s = P[P.kind == "data"].drop_duplicates(["sub", "ses", "task"])
    both = s[s.ses.isin(["ses-1", "ses-2"])].groupby(["sub", "task"]).ses.nunique()
    rs = both[both == 2].reset_index().merge(m1, on="sub", how="left")
    t["two sessions"] = {task: dict(participants=int(len(g)), healthy=int((g.indication == "HEALTHY").sum()),
                                    under_18=int((g.age < 18).sum()), age_range=[float(g.age.min()), float(g.age.max())],
                                    indications={k: int(v) for k, v in g.indication.value_counts().items()})
                         for task, g in rs.groupby("task")}
    out["TDBRAIN"] = t
    for name, f, unit in (("Sleep-EDF", "results/post_sleep_sigma.csv", "rec"), ("ANPHY-Sleep", "results/post_anphy_sigma.csv", "rec")):
        S = pd.read_csv(f, usecols=["rec", "ch", "stage", "chunk", "kind"])
        dat, tw = S[S.kind == "data"], S[S.kind == "iaaft0"]
        e = dict(recordings=int(dat.rec.nunique()), channels=int(dat.ch.nunique()),
                 pieces_per_stage={st: int(n) for st, n in dat.drop_duplicates(["rec", "stage", "chunk"]).stage.value_counts().items()},
                 recordings_per_stage=stage_counts(dat, "rec"),
                 channel_pieces=int(len(dat)), channel_pieces_with_twin=int(len(tw)))
        if name == "Sleep-EDF":
            e["participants"] = int(dat.rec.str[:5].nunique())
            e["participants_per_stage"] = {st: int(n) for st, n in dat.assign(p=dat.rec.str[:5]).groupby("stage").p.nunique().items()}
        out[name] = e
    for name, f in (("Mumtaz", "results/post_mumtaz.csv"), ("ds003478", "results/post_ds003478.csv")):
        S = pd.read_csv(f, usecols=lambda c: c in ("sub", "ch", "task", "kind", "group", "BDI", "chunk"))
        S = S[S.task == "restEC"] if "task" in S else S
        out[name] = dict(participants=int(S["sub"].nunique()), channel_pieces=int(len(S)), kinds=S.kind.value_counts().to_dict())
    json.dump(out, open("results/population_counts.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

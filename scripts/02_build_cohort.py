"""Step 2: cohort, 24-month label, participant split, leakage checks, model inputs."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mdp import cohort as C  # noqa: E402
from mdp import features as F  # noqa: E402
from mdp.config import PROCESSED, RUNS, TABLES, data_csv, ensure_dirs, load_config  # noqa: E402


def describe(cohort, visits, raw):
    """Table: characteristics of the development and test partitions."""
    d = cohort.copy()
    d["group"] = np.where(d.split == "test", "Locked test", "Development (train+val+cal)")
    d["age"] = raw["static"][:, 0]
    d["female"] = raw["static"][:, 1]
    d["educ"] = raw["static"][:, 2]
    d["apoe4_carrier"] = np.where(np.isnan(raw["static"][:, 3]), np.nan, raw["static"][:, 3] > 0)
    for s in C.SCORES:
        d[s] = visits.loc[d.index_row, s].to_numpy()
    d["visits"] = raw["lengths"]
    d["mri"] = d.mri_row >= 0

    def ms(x):
        return f"{np.nanmean(x):.1f} ({np.nanstd(x):.1f})"

    rows = []
    for name, g in d.groupby("group", sort=False):
        rows.append({
            "Partition": name, "Participants": g.RID.nunique(), "Index windows": len(g),
            "Converters, n (%)": f"{int(g.label.sum())} ({g.label.mean():.1%})",
            "Age, mean (SD)": ms(g.age), "Female, %": f"{np.mean(g.female):.1%}",
            "Education, mean (SD)": ms(g.educ), "APOE4 carrier, %": f"{np.nanmean(g.apoe4_carrier):.1%}",
            "MMSE, mean (SD)": ms(g.MMSE), "CDR-SB, mean (SD)": ms(g.CDRSB),
            "ADAS-Cog13, mean (SD)": ms(g.ADAS13), "FAQ, mean (SD)": ms(g.FAQ),
            "Visits in history, mean (SD)": ms(g.visits), "MRI available, %": f"{g.mri.mean():.1%}",
        })
    return pd.DataFrame(rows).set_index("Partition").T


def main():
    cfg = load_config()
    ensure_dirs()
    visits, flow = C.load_visits(data_csv(cfg))
    cohort, flow2, linkage = C.build_cohort(visits, cfg)

    manifest = C.split_participants(cohort, cfg, cfg["seed"])
    cohort = cohort.merge(manifest, on="RID")
    C.assert_no_participant_overlap(cohort)
    C.assert_predictors_before_cutoff(cohort, visits)
    print("Leakage checks passed: no participant in two splits; all predictors precede the cut-off.")

    raw = F.extract_raw(visits, cohort, cfg["cohort"]["max_history_visits"])
    scalers = F.fit_scalers(raw, (cohort.split == "train").to_numpy())
    arrays = F.transform(raw, scalers)
    tab, clinical_cols = F.summary_features(raw)

    cohort.to_csv(PROCESSED / "cohort.csv", index=False)
    np.savez(PROCESSED / "features.npz", **arrays)
    tab.to_csv(PROCESSED / "tabular.csv", index=False)
    with open(PROCESSED / "feature_meta.json", "w") as f:
        json.dump({"clinical_cols": clinical_cols, "scalers": {k: np.asarray(v).tolist() for k, v in scalers.items()}},
                  f, indent=1)

    flow_df = pd.DataFrame([{"Stage": s, "Count": n, "Participants": p} for s, n, p in flow + flow2])
    flow_df.to_csv(TABLES / "table_cohort_flow.csv", index=False)
    pd.DataFrame([linkage]).to_csv(TABLES / "table_mri_linkage.csv", index=False)
    describe(cohort, visits, raw).to_csv(TABLES / "table_descriptive.csv")

    split_counts = cohort.groupby("split").agg(participants=("RID", "nunique"), windows=("window_id", "size"),
                                               converters=("label", "sum"))
    split_counts.to_csv(TABLES / "table_split_counts.csv")
    with open(RUNS / "split_info.json", "w") as f:
        json.dump({"split_manifest_sha256_16": C.manifest_hash(manifest), "seed": cfg["seed"],
                   "data_source": cfg["data"]["source"]}, f, indent=1)

    print(flow_df.to_string(index=False))
    print("MRI linkage:", linkage)
    print(split_counts)


if __name__ == "__main__":
    main()

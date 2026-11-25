"""Step 4b: why conformal coverage fell short - compares the calibration split with the test sets."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mdp import metrics as M  # noqa: E402
from mdp.config import PREDICTIONS, PROCESSED, RESULTS, ROOT, TABLES, data_csv, load_config  # noqa: E402


def main():
    cfg = load_config()
    T = json.load(open(RESULTS / "metrics_summary.json"))["temperatures"]["e5_proposed"]
    pred = pd.read_csv(PREDICTIONS / "e5_proposed.csv.gz")
    logit = pred[pred.condition == "natural"].groupby(["cohort", "window_id"]).logit.mean()
    dev = pd.read_csv(PROCESSED / "cohort.csv")
    sets = {"Calibration split": ("dev", dev[dev.split == "calibration"], data_csv(cfg)),
            "Development test": ("dev", dev[dev.split == "test"], data_csv(cfg))}
    for name in ("external_A", "external_B"):
        sets[name.replace("_", " ").title()] = (name, pd.read_csv(PROCESSED / f"cohort_{name}.csv"),
                                                ROOT / cfg["data"]["external"][name]["csv"])
    rows = []
    for label, (cohort, c, csv) in sets.items():
        dx_bl = pd.read_csv(csv, usecols=["RID", "DX_bl"]).groupby("RID").DX_bl.first()
        p = M.sigmoid(logit.loc[cohort].reindex(c.window_id).to_numpy() / T)
        neg = c.label.to_numpy() == 0
        rows.append({"Set": label, "Participants": c.RID.nunique(), "Windows": len(c),
                     "Conversion rate": round(c.label.mean(), 3),
                     "Late MCI at entry (share of windows)": round((c.RID.map(dx_bl) == "LMCI").mean(), 3),
                     "Non-converters: mean predicted risk": round(float(p[neg].mean()), 3),
                     "Non-converters: 90th percentile of predicted risk": round(float(np.quantile(p[neg], 0.9)), 3)})
    pd.DataFrame(rows).to_csv(TABLES / "table_conformal_diagnostic.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()

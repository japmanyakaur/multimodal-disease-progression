"""Step 1: write the simulated ADNIMERGE-format cohort to data/synthetic/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mdp.config import ROOT, load_config  # noqa: E402
from mdp.simulate import simulate  # noqa: E402


def main():
    cfg = load_config()
    visits, truth = simulate(cfg["data"]["n_participants"], cfg["seed"])
    out = ROOT / cfg["data"]["synthetic_csv"]
    out.parent.mkdir(parents=True, exist_ok=True)
    visits.to_csv(out, index=False)
    truth.to_csv(out.parent / "simulation_truth.csv", index=False)

    print(f"Wrote {len(visits)} visits for {visits.RID.nunique()} participants -> {out}")
    print("Baseline groups:", visits.groupby("RID").DX_bl.first().value_counts().to_dict())
    print("Diagnoses over all visits:", visits.DX.value_counts().to_dict())
    print(f"Visits with usable MRI: {visits.Hippocampus.notna().mean():.1%}")


if __name__ == "__main__":
    main()

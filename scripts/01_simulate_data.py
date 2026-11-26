"""Step 1: write the simulated ADNIMERGE-format development cohort and the two external test cohorts."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mdp.config import ROOT, load_config  # noqa: E402
from mdp.simulate import simulate  # noqa: E402


def write(visits, truth, out, truth_out, name):
    out.parent.mkdir(parents=True, exist_ok=True)
    visits.to_csv(out, index=False)
    truth.to_csv(truth_out, index=False)
    print(f"{name}: {len(visits)} visits, {visits.RID.nunique()} participants, "
          f"usable MRI at {visits.Hippocampus.notna().mean():.1%} of visits -> {out.name}")


def main():
    cfg = load_config()
    n = cfg["data"]["n_participants"]
    visits, truth = simulate(n, cfg["seed"])
    out = ROOT / cfg["data"]["synthetic_csv"]
    write(visits, truth, out, out.parent / "simulation_truth.csv", "Development cohort")

    for name, ext in cfg["data"]["external"].items():
        v, t = simulate(n, ext["seed"], qc_probs=tuple(ext["qc_probs"]),
                        mri_acquisition_multiplier=ext["mri_acquisition_multiplier"])
        csv = ROOT / ext["csv"]
        write(v, t, csv, csv.parent / f"simulation_truth_{name}.csv", name)


if __name__ == "__main__":
    main()

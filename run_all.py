"""Run the whole pipeline: data -> cohort -> training -> evaluation -> figures.

Usage:  python run_all.py            (about 20 minutes on a 4-core laptop CPU, mostly the 3D CNN)
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = [
    ("1/7 Simulate ADNIMERGE-format data (development + external cohorts)", "scripts/01_simulate_data.py"),
    ("2/7 Build cohorts, split, leakage checks, audit, 3D volumes", "scripts/02_build_cohort.py"),
    ("3/7 Hyper-parameter search and training (E1-E5, ablations, regression)", "scripts/03_train_models.py"),
    ("4/7 3D CNN, CNN fusion models and Grad-CAM", "scripts/03b_train_imaging.py"),
    ("5/7 Evaluate (hypotheses, E6, E7, external cohorts)", "scripts/04_evaluate.py"),
    ("6/7 Make figures", "scripts/05_make_figures.py"),
    ("7/7 Dataset card and model card", "scripts/06_make_cards.py"),
]


def main():
    skip_data = "--use-existing-data" in sys.argv
    for title, script in STEPS:
        if skip_data and script.startswith("scripts/01"):
            continue
        print(f"\n=== {title} ===", flush=True)
        start = time.time()
        subprocess.run([sys.executable, str(ROOT / script)], check=True, cwd=ROOT)
        print(f"--- finished in {time.time() - start:.0f}s", flush=True)
    print("\nAll steps complete. Tables: results/tables  Figures: results/figures")


if __name__ == "__main__":
    main()

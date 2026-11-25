"""Run the whole pipeline: data -> cohort -> training -> evaluation -> figures.

Usage:  python run_all.py            (about 3-4 minutes on a 4-core laptop CPU)
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = [
    ("1/5 Simulate ADNIMERGE-format data", "scripts/01_simulate_data.py"),
    ("2/5 Build cohort, split, leakage checks", "scripts/02_build_cohort.py"),
    ("3/5 Train all models (E1-E5, ablations)", "scripts/03_train_models.py"),
    ("4/5 Evaluate (E6, E7, hypotheses)", "scripts/04_evaluate.py"),
    ("5/5 Make figures", "scripts/05_make_figures.py"),
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

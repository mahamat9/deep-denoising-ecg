import os
import subprocess
import sys
from datetime import datetime


def _run_step(step_name: str, script_path: str, log_file) -> None:
    log_file.write(f"\n=== {step_name} ===\n")
    log_file.flush()

    process = subprocess.Popen(
        [sys.executable, script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    for line in process.stdout:
        print(line, end="")
        log_file.write(line)
    return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(f"Step '{step_name}' failed with exit code {return_code}.")


def run_full_pipeline():
    results_dir = os.path.join(".", "results")
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "result.out")

    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"Run started: {datetime.now().isoformat()}\n")
        _run_step("Step 1: Data Setup", "data_setup.py", log_file)
        _run_step("Step 2: Training Autoencoder", os.path.join("scripts", "train_autoencoder.py"), log_file)
        _run_step("Step 3: Training Diffusion Model", os.path.join("scripts", "train_diffusion.py"), log_file)
        _run_step("Step 4: Final Evaluation", os.path.join("scripts", "evaluate.py"), log_file)
        log_file.write(f"Run completed: {datetime.now().isoformat()}\n")

#.sh
"""
python3 data_setup.py
python3 scripts/train_autoencoder.py > log_autoencoder_train.out
python3 scripts/train_diffusion.py > log_diffusion_train.out
python3 scripts/evaluate.py > log_eval.out
"""
if __name__ == "__main__":
    run_full_pipeline()
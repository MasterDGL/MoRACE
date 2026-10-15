from pathlib import Path
import subprocess
import sys


def test_projection_contract_harness_runs_in_clean_interpreter():
    project_root = Path(__file__).resolve().parents[1]
    harness = project_root / "tests" / "projection_contract_harness.py"
    completed = subprocess.run(
        [sys.executable, str(harness), "-v"],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, (
        "Projection contract harness failed in its isolated interpreter.\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )

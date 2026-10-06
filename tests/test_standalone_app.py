"""
tests/test_standalone_app.py

Automated tests to verify that the packaged standalone Windows executable
launches and initializes without missing DLLs or import failures.
"""

import os
import subprocess
import sys
import time
from pathlib import Path
import pytest


def test_packaged_executable_starts():
    """Verifies that the compiled standalone executable can launch and run check/help."""
    repo_root = Path(__file__).resolve().parent.parent
    exe_path = repo_root / "dist" / "deepfeik.exe"
    
    # Check if executable exists
    if not exe_path.exists():
        pytest.fail(f"Packaged executable not found at {exe_path}. Build step failed to produce binary.")
    
    assert exe_path.stat().st_size > 5 * 1024 * 1024, "Executable is suspiciously small (< 5 MB)"

    # Redirect to files instead of anonymous pipes to prevent Windows IPC deadlocks on frozen GUI executables
    stdout_file = repo_root / "dist" / "test_stdout.log"
    stderr_file = repo_root / "dist" / "test_stderr.log"
    with open(stdout_file, "w", encoding="utf-8") as out_f, open(stderr_file, "w", encoding="utf-8") as err_f:
        res = subprocess.run(
            [str(exe_path), "--check-install"],
            stdout=out_f,
            stderr=err_f,
            timeout=60,
        )
    
    err_text = stderr_file.read_text(encoding="utf-8", errors="replace") if stderr_file.exists() else ""
    out_text = stdout_file.read_text(encoding="utf-8", errors="replace") if stdout_file.exists() else ""

    assert res.returncode == 0, (
        f"Executable failed with returncode {res.returncode}. "
        f"Stderr: {err_text.strip()}. Stdout: {out_text.strip()}"
    )
    print("Standalone executable verified: started cleanly and validated installation!")


if __name__ == "__main__":
    test_packaged_executable_starts()

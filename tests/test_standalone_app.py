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

    # Test launching with --check-install
    res = subprocess.run([str(exe_path), "--check-install"], capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, f"Executable failed with code {res.returncode}. Stderr: {res.stderr}. Stdout: {res.stdout}"
    print("Standalone executable verified: started cleanly and validated installation!")


if __name__ == "__main__":
    test_packaged_executable_starts()

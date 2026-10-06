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


def test_packaged_executable_starts():
    """Verifies that the compiled standalone executable can launch and run help/check."""
    repo_root = Path(__file__).resolve().parent.parent
    exe_path = repo_root / "dist" / "deepfeik.exe"
    
    # Check if executable exists
    assert exe_path.exists(), f"Packaged executable not found at {exe_path}"
    assert exe_path.stat().st_size > 5 * 1024 * 1024, "Executable is suspiciously small (< 5 MB)"

    # Test launching with --help
    res = subprocess.run([str(exe_path), "--help"], capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, f"Executable failed with code {res.returncode}. Stderr: {res.stderr}"
    assert "usage:" in res.stdout.lower() or "deepfeik" in res.stdout.lower()
    print("Standalone executable verified: started cleanly and returned help text!")


if __name__ == "__main__":
    test_packaged_executable_starts()

import subprocess
import sys
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parents[1]


def test_standalone_run_never_binds_to_real_dev_db():
    """A pytest invocation that only touches a test file with no DB isolation
    of its own must still never bind db_engine to the real dev database
    (backend/store/meta.sqlite) -- regardless of which file pytest happens
    to import first in that process."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_upload_endpoint_async.py",
         "--collect-only", "-q", "--log-cli-level=INFO"],
        cwd=_BACKEND_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    output = result.stdout + result.stderr
    assert "Created SQLite engine" in output, f"engine creation log not found:\n{output}"
    assert "./store/meta.sqlite" not in output, (
        "db_engine bound to the real dev database during an isolated test run:\n" + output
    )

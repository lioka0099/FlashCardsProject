import os
import tempfile
from pathlib import Path

# Runs before pytest imports any test module in this directory, regardless of
# which file(s) are selected or their collection order -- unlike a per-file
# `os.environ["DATABASE_URL"] = ...` line, which only takes effect if that
# file happens to be the first thing to import app.data.db_engine in this
# process. Without this, a test file with no isolation of its own can bind
# db_engine's module-level engine to the real local dev database.
_TEST_DB_ROOT = Path(tempfile.mkdtemp(prefix="flashcards_test_"))
os.environ["DATABASE_URL"] = f"sqlite:///{(_TEST_DB_ROOT / 'meta.sqlite').as_posix()}"

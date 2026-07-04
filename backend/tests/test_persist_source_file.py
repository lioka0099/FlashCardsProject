import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app.services.ingestion import _persist_source_file


class PersistSourceFileTests(unittest.TestCase):
    def test_copies_file_to_doc_id_path_and_survives_original_deletion(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            src = tmp_path / "Chapter 2.pdf"
            src.write_bytes(b"%PDF-1.4 fake")
            dest_dir = tmp_path / "documents"

            persisted = _persist_source_file(str(src), "abc123def456", dest_dir=dest_dir)

            # Simulate the worker deleting the temp upload.
            os.remove(src)

            self.assertTrue(Path(persisted).exists())
            self.assertEqual(Path(persisted).name, "abc123def456.pdf")
            self.assertEqual(Path(persisted).read_bytes(), b"%PDF-1.4 fake")

    def test_returns_absolute_path(self):
        with TemporaryDirectory() as tmp:
            src = Path(tmp) / "notes.txt"
            src.write_text("hello")
            persisted = _persist_source_file(str(src), "doc999", dest_dir=Path(tmp) / "documents")
            self.assertTrue(Path(persisted).is_absolute())


if __name__ == "__main__":
    unittest.main()

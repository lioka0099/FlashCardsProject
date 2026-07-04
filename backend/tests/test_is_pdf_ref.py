import unittest

from app.api.endpoints import _is_pdf_ref


class IsPdfRefTests(unittest.TestCase):
    def test_local_pdf_path(self):
        self.assertTrue(_is_pdf_ref("/Users/x/uploads/documents/abc123.pdf"))

    def test_uppercase_extension(self):
        self.assertTrue(_is_pdf_ref("/tmp/Slides.PDF"))

    def test_http_url_with_query_and_hash(self):
        self.assertTrue(_is_pdf_ref("https://example.com/source.pdf?token=1#page=3"))

    def test_docx_is_not_pdf(self):
        self.assertFalse(_is_pdf_ref("/tmp/notes.docx"))

    def test_txt_is_not_pdf(self):
        self.assertFalse(_is_pdf_ref("/tmp/notes.txt"))

    def test_bare_doc_id_without_extension_is_not_pdf(self):
        self.assertFalse(_is_pdf_ref("88922247aea5"))

    def test_empty_is_not_pdf(self):
        self.assertFalse(_is_pdf_ref(""))


if __name__ == "__main__":
    unittest.main()

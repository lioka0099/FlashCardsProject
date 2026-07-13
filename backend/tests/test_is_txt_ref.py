import unittest

from app.api.endpoints import _is_txt_ref


class IsTxtRefTests(unittest.TestCase):
    def test_local_txt_path(self):
        self.assertTrue(_is_txt_ref("/Users/x/uploads/documents/abc123.txt"))

    def test_uppercase_extension(self):
        self.assertTrue(_is_txt_ref("/tmp/Notes.TXT"))

    def test_http_url_with_query_and_hash(self):
        self.assertTrue(_is_txt_ref("https://example.com/source.txt?token=1#page=3"))

    def test_pdf_is_not_txt(self):
        self.assertFalse(_is_txt_ref("/tmp/slides.pdf"))

    def test_docx_is_not_txt(self):
        self.assertFalse(_is_txt_ref("/tmp/notes.docx"))

    def test_bare_doc_id_without_extension_is_not_txt(self):
        self.assertFalse(_is_txt_ref("88922247aea5"))

    def test_empty_is_not_txt(self):
        self.assertFalse(_is_txt_ref(""))


if __name__ == "__main__":
    unittest.main()

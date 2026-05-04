from pathlib import Path
from typing import List, TypedDict, Union

class Page(TypedDict):
    page: int
    text: str

def load_pdf(path: Union[str, Path]) -> List[Page]:
    """
    Robust PDF text extraction:
    1) Try pdfminer.six
    2) Try PyPDF
    3) Fallback to PyMuPDF (fitz) for better layout fidelity
    """
    # 1) Try pdfminer.six
    try:
        from pdfminer.high_level import extract_text  # type: ignore
        # pdfminer extracts whole doc; we split by form feed which often denotes pages
        full_text = extract_text(str(path)) or ""
        if full_text:
            parts = [p for p in full_text.split("\f")]
            pages: List[Page] = [{"page": i + 1, "text": t} for i, t in enumerate(parts)]
            if not pages:
                pages = [{"page": 1, "text": full_text}]
            return pages
    except Exception:
        pass

    # 2) Try PyPDF before PyMuPDF because PyMuPDF can segfault on some PDFs.
    try:
        from pypdf import PdfReader  # type: ignore
        reader = PdfReader(str(path))
        pages: List[Page] = []
        for i, p in enumerate(reader.pages, start=1):
            pages.append({"page": i, "text": p.extract_text() or ""})
        if pages:
            return pages
    except Exception:
        pass

    # 3) Try PyMuPDF
    try:
        import fitz  # type: ignore
        pages: List[Page] = []
        with fitz.open(str(path)) as doc:
            for i in range(len(doc)):
                page = doc.load_page(i)
                text = page.get_text("text") or ""
                pages.append({"page": i + 1, "text": text})
        if any(p["text"].strip() for p in pages):
            return pages
    except Exception:
        pass

    raise ImportError(
        "No PDF extractors available. Install one of: 'pip install pdfminer.six' or 'pip install pypdf' or 'pip install pymupdf'."
    )

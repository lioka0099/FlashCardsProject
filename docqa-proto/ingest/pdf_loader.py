from pathlib import Path
from typing import List, TypedDict, Union

class Page(TypedDict):
    page: int
    text: str

def load_pdf(path: Union[str, Path]) -> List[Page]:
    """
    Robust PDF text extraction:
    1) Try PyMuPDF (fitz) for better layout fidelity
    2) Fallback to pdfminer.six
    3) Fallback to PyPDF
    """
    # 1) Try PyMuPDF
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

    # 2) Try pdfminer.six
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

    # 3) Fallback to PyPDF
    try:
        from pypdf import PdfReader  # type: ignore
        reader = PdfReader(str(path))
        pages: List[Page] = []
        for i, p in enumerate(reader.pages, start=1):
            pages.append({"page": i, "text": p.extract_text() or ""})
        return pages
    except Exception as exc:
        raise ImportError(
            "No PDF extractors available. Install one of: 'pip install pymupdf' or 'pip install pdfminer.six' or 'pip install pypdf'."
        ) from exc

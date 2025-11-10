from pathlib import Path
from typing import List, TypedDict, Union

class Page(TypedDict):
    page: int
    text: str

def load_pdf(path: Union[str, Path]) -> List[Page]:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception as exc:
        raise ImportError(
            "PyPDF is required but not installed. Install it with 'pip install pypdf'."
        ) from exc
    reader = PdfReader(str(path))
    pages: List[Page] = []
    for i, p in enumerate(reader.pages, start=1):
        pages.append({"page": i, "text": p.extract_text() or ""})
    return pages

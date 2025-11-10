from typing import Any, Dict, List, TypedDict

class Chunk(TypedDict):
    doc_id: str
    page: int
    chunk_id: str
    text: str
    start: int
    end: int

def _split_paras(text: str) -> List[str]:
    raw = [p.strip() for p in text.split("\n") if p.strip()]
    paras: List[str] = []
    buf = []
    for p in raw:
        if len(p) < 300:
            buf.append(p)
        else:
            if buf: paras.append(" ".join(buf)); buf = []
            paras.append(p)
    if buf: paras.append(" ".join(buf))
    return paras

def make_chunks(doc_id: str, pages: List[Dict[str, Any]],
                target_chars: int = 500, overlap: int = 100) -> List[Chunk]:
    chunks: List[Chunk] = []
    for page_obj in pages:
        page = int(page_obj["page"])
        text = page_obj["text"] or ""
        paras = _split_paras(text)

        # accumulate into windows near target_chars
        cursor = 0
        buf, buf_len, buf_start = [], 0, 0
        for para in paras + [""]:
            if para and (buf_len + len(para) < target_chars):
                if not buf: buf_start = cursor
                buf.append(para)
                buf_len += len(para) + 1
                cursor += len(para) + 1
                continue

            if buf:
                body = "\n".join(buf).strip()
                start = buf_start
                end = start + len(body)
                chunk_id = f"{doc_id}-{page}-{len(chunks)}"
                chunks.append({
                    "doc_id": doc_id, "page": page, "chunk_id": chunk_id,
                    "text": body, "start": start, "end": end
                })
                # overlap: rewind some chars
                cursor = max(0, end - overlap)
                # re-seed next window from text slice after overlap
                remainder = text[cursor:end]  # unused, just conceptual
                buf, buf_len = [], 0

            # start next with current para (if not the dummy "")
            if para:
                buf = [para]
                buf_len = len(para) + 1
                buf_start = cursor
                cursor += len(para) + 1
    return chunks

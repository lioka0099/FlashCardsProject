from typing import Any, Dict, List, TypedDict

class Chunk(TypedDict):
    doc_id: str
    page: int
    chunk_id: str
    text: str
    start: int
    end: int

def _split_long_text(text: str, max_len: int) -> List[str]:
    segments: List[str] = []
    i, n = 0, len(text)
    while i < n:
        end = min(n, i + max_len)
        if end < n:
            cut = text.rfind(" ", i, end)
            if cut == -1 or cut <= i + max_len // 3:
                cut = end
        else:
            cut = end
        segment = text[i:cut].strip()
        if segment:
            segments.append(segment)
        i = cut
        while i < n and text[i].isspace():
            i += 1
    return segments or [text[:max_len]]

def _split_paras(text: str, max_len: int) -> List[str]:
    raw = [p.strip() for p in text.split("\n") if p.strip()]
    paras: List[str] = []
    buf: List[str] = []
    for p in raw:
        if len(p) < min(300, max_len):
            buf.append(p)
            continue
        if buf:
            joined = " ".join(buf).strip()
            if joined:
                if len(joined) > max_len:
                    paras.extend(_split_long_text(joined, max_len))
                else:
                    paras.append(joined)
            buf = []
        if len(p) > max_len:
            paras.extend(_split_long_text(p, max_len))
        else:
            paras.append(p)
    if buf:
        joined = " ".join(buf).strip()
        if joined:
            if len(joined) > max_len:
                paras.extend(_split_long_text(joined, max_len))
            else:
                paras.append(joined)
    if not paras and raw:
        for piece in _split_long_text(" ".join(raw), max_len):
            paras.append(piece)
    return paras

def make_chunks(doc_id: str, pages: List[Dict[str, Any]],
                target_chars: int = 600, overlap: int = 120) -> List[Chunk]:
    chunks: List[Chunk] = []
    for page_obj in pages:
        page = int(page_obj["page"])
        text = page_obj["text"] or ""
        paras = _split_paras(text, max_len=max(200, target_chars))

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

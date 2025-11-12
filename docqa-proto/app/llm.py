from typing import List, Tuple
import numpy as np
from app.utils import getenv

_client = None
def client():
    # Lazy import to avoid hard dependency during unrelated code paths.
    try:
        from openai import OpenAI  # type: ignore
    except Exception as exc:
        raise ImportError(
            "OpenAI SDK is required but not installed. Install it with 'pip install openai'."
        ) from exc
    global _client
    if _client is None:
        _client = OpenAI(api_key=getenv("OPENAI_API_KEY"))
    return _client

EMBED_MODEL = getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = getenv("CHAT_MODEL", "gpt-4o-mini")

def embed_texts(texts: List[str]) -> np.ndarray:
    # OpenAI returns list of vectors
    resp = client().embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = [d.embedding for d in resp.data]
    return np.array(vecs, dtype="float32")

def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])

def generate_alternate_queries(question: str, num_variations: int = 3) -> List[str]:
    """
    Use the chat model to produce semantically diverse reformulations of the query
    to improve recall (multi-query retrieval).
    """
    prompt = (
        "Rewrite the user's question into diverse, concise search queries that capture different phrasings.\n"
        f"Original question: {question}\n"
        f"Return exactly {num_variations} queries, one per line, no numbering."
    )
    resp = client().chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "You are an expert search assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
        max_tokens=120,
    )
    text = resp.choices[0].message.content or ""
    queries = [q.strip() for q in text.splitlines() if q.strip()]
    if len(queries) < num_variations:
        # pad with original question if the model returned fewer
        queries += [question] * (num_variations - len(queries))
    return queries[:num_variations]

def rerank_chunks(question: str, candidates: List[Tuple[str, str]], top_n: int = 8) -> List[Tuple[int, float]]:
    """
    Rerank candidate chunks with the chat model.
    Input: list of (chunk_id, snippet) strings.
    Output: list of (index_in_input, score) sorted by descending score.
    """
    if not candidates:
        return []
    # Build a compact list with indices to keep the response small and structured
    lines = []
    for i, (cid, text) in enumerate(candidates):
        preview = text.replace("\n", " ")
        if len(preview) > 600:
            preview = preview[:600] + "..."
        lines.append(f"{i}\t{cid}\t{preview}")
    instruction = (
        "You are a ranking model. Score each candidate chunk for relevance to the question from 0.0 to 1.0.\n"
        "Return a JSON array of objects with fields: index (int), score (float). No extra text."
    )
    user = (
        f"QUESTION:\n{question}\n\n"
        "CANDIDATES (index<TAB>chunk_id<TAB>text):\n" + "\n".join(lines)
    )
    resp = client().chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": user},
        ],
        temperature=0,
        max_tokens=400,
        response_format={"type": "json_object"},  # requires newer OpenAI SDK; safe to keep as hint
    )
    import json
    raw = resp.choices[0].message.content or "[]"
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "results" in data:
            arr = data["results"]
        else:
            arr = data if isinstance(data, list) else []
    except Exception:
        arr = []
    scored: List[Tuple[int, float]] = []
    for obj in arr:
        try:
            scored.append((int(obj["index"]), float(obj["score"])))
        except Exception:
            continue
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]

try:
	from fastapi import FastAPI, UploadFile, File, Form  # type: ignore
	from fastapi.responses import JSONResponse  # type: ignore
except Exception as exc:
	raise ImportError(
		"FastAPI is required but not installed. Install it with 'pip install fastapi uvicorn'."
	) from exc
from pathlib import Path
import shutil, tempfile

from app.api import ingest_document
from app.answer import generate_answer

app = FastAPI(title="DocQA Proto")

@app.post("/ingest")
async def ingest_endpoint(file: UploadFile = File(...)):
    # Save uploaded file to a temp path inside ./uploads
    uploads = Path("uploads"); uploads.mkdir(exist_ok=True)
    temp_path = uploads / file.filename
    with temp_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    res = ingest_document(str(temp_path))
    return {"ok": True, "doc_id": res.doc_id, "num_chunks": res.num_chunks, "filename": file.filename}

@app.post("/ask")
async def ask_endpoint(question: str = Form(...), k: int = Form(5), min_score: float = Form(0.35)):
    ans = generate_answer(question=question, k=k, min_score=min_score)
    # Format proofs for response (short text)
    proofs = [{
        "doc_id": p.doc_id, "page": p.page, "score": p.score,
        "start": p.start, "end": p.end, "text": p.text[:300] + ("..." if len(p.text) > 300 else "")
    } for p in ans.proofs]
    return JSONResponse({"ok": True, "answer": ans.answer, "proofs": proofs})

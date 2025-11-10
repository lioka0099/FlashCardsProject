try:
    from pydantic import BaseModel  # type: ignore
except Exception:
    # Minimal fallback shim to avoid hard dependency during prototyping.
    class BaseModel:  # type: ignore
        def __init__(self, **data):
            for k, v in data.items():
                setattr(self, k, v)
        def model_dump(self):
            if hasattr(self, "__annotations__"):
                return {k: getattr(self, k) for k in self.__annotations__.keys()}
            return dict(self.__dict__)

class IngestResult(BaseModel):
    doc_id: str
    num_chunks: int

class ProofSpan(BaseModel):
    doc_id: str
    page: int
    start: int
    end: int
    text: str
    score: float

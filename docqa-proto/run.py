import argparse, json
from dotenv import load_dotenv
load_dotenv()

from app.api import ingest_document, retrieve_with_proofs
from store.storage import VectorStore
from app.answer import generate_answer

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ingest", help="path to document to ingest")
    p.add_argument("--ask", help="question: retrieve proofs only (no LLM answer)")
    p.add_argument("--answer", help="question: retrieve + generate answer with citations")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--min_score", type=float, default=0.5)
    args = p.parse_args()

    store = VectorStore()

    if args.ingest:
        res = ingest_document(args.ingest, store)
        print(f"Ingested doc_id={res.doc_id} chunks={res.num_chunks}")

    if args.ask:
        proofs = retrieve_with_proofs(args.ask, k=args.k, store=store)
        print(json.dumps([p.model_dump() for p in proofs], ensure_ascii=False, indent=2))

    if args.answer:
        ans = generate_answer(args.answer, k=args.k, min_score=args.min_score)
        print("\n=== ANSWER ===\n")
        print(ans.answer)
        print("\n=== PROOFS ===\n")
        for i, p in enumerate(ans.proofs, 1):
            print(f"S{i} | doc={p.doc_id} page={p.page} score={p.score:.2f}")
            print(p.text[:300] + ("..." if len(p.text) > 300 else ""))
            print()

if __name__ == "__main__":
    main()

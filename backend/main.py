"""Legacy command-line entrypoint for the flashcards pipeline (--ingest / --create_exam / --demo).
Separate from the FastAPI app in app/api; kept for manual and local runs."""

import argparse, json, logging, mimetypes, time, sys
from pathlib import Path
from typing import Any, Dict, Optional
import re

import httpx
from dotenv import load_dotenv
load_dotenv()

_ANSI_RE = re.compile(r"\x1B\[[0-9;]*[A-Za-z]")

class TeeTextOutput:
    """Write console output to terminal and plain-text file."""
    def __init__(self, console_stream, file_stream):
        self.console_stream = console_stream
        self.file_stream = file_stream

    def write(self, text: str):
        self.console_stream.write(text)
        self.file_stream.write(_ANSI_RE.sub("", text))
        return len(text)

    def flush(self):
        self.console_stream.flush()
        self.file_stream.flush()

    def isatty(self):
        return self.console_stream.isatty()

# ═══════════════════════════════════════════════════════════════════════════════
# PRESENTATION UTILITIES - Beautiful Terminal Output
# ═══════════════════════════════════════════════════════════════════════════════

class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'
    
    @classmethod
    def disable(cls):
        """Disable colors for non-supporting terminals"""
        cls.HEADER = cls.BLUE = cls.CYAN = cls.GREEN = ''
        cls.YELLOW = cls.RED = cls.BOLD = cls.DIM = cls.RESET = ''

# Check if terminal supports colors
if sys.platform == 'win32':
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except:
        Colors.disable()

def print_banner():
    """Print beautiful ASCII art banner"""
    banner = f"""
{Colors.CYAN}╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  {Colors.BOLD}{Colors.YELLOW}    ███████╗██╗      █████╗ ███████╗██╗  ██╗ ██████╗ █████╗ ██████╗ ██████╗   {Colors.CYAN}║
║  {Colors.BOLD}{Colors.YELLOW}    ██╔════╝██║     ██╔══██╗██╔════╝██║  ██║██╔════╝██╔══██╗██╔══██╗██╔══██╗  {Colors.CYAN}║
║  {Colors.BOLD}{Colors.YELLOW}    █████╗  ██║     ███████║███████╗███████║██║     ███████║██████╔╝██║  ██║  {Colors.CYAN}║
║  {Colors.BOLD}{Colors.YELLOW}    ██╔══╝  ██║     ██╔══██║╚════██║██╔══██║██║     ██╔══██║██╔══██╗██║  ██║  {Colors.CYAN}║
║  {Colors.BOLD}{Colors.YELLOW}    ██║     ███████╗██║  ██║███████║██║  ██║╚██████╗██║  ██║██║  ██║██████╔╝  {Colors.CYAN}║
║  {Colors.BOLD}{Colors.YELLOW}    ╚═╝     ╚══════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝   {Colors.CYAN}║
║                                                                              ║
║  {Colors.DIM}         AI-Powered Flashcard Generation System                            {Colors.CYAN}║
║  {Colors.DIM}         Intelligent Learning Through Document Analysis                    {Colors.CYAN}║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝{Colors.RESET}
"""
    print(banner)

def print_step_header(step_num, total_steps, title, icon=""):
    """Print a beautiful step header"""
    progress = "█" * step_num + "░" * (total_steps - step_num)
    print(f"""
{Colors.CYAN}┌──────────────────────────────────────────────────────────────────────────────┐
│  {Colors.BOLD}{Colors.YELLOW}{icon} STEP {step_num}/{total_steps}: {title.upper():<55}{Colors.CYAN}│
│  {Colors.DIM}[{progress}] {int(step_num/total_steps*100):>3}%{Colors.CYAN}                                                    │
└──────────────────────────────────────────────────────────────────────────────┘{Colors.RESET}
""")

def print_success(message):
    """Print success message"""
    print(f"  {Colors.GREEN}✓{Colors.RESET} {message}")

def print_info(message):
    """Print info message"""
    print(f"  {Colors.CYAN}ℹ{Colors.RESET} {message}")

def print_item(message, indent=2):
    """Print list item"""
    spaces = " " * indent
    print(f"{spaces}{Colors.DIM}•{Colors.RESET} {message}")

def print_section_divider():
    """Print a section divider"""
    print(f"\n{Colors.DIM}{'─' * 80}{Colors.RESET}\n")

def print_card(card_num, topic, question, answer, proofs):
    """Print a beautifully formatted flashcard"""
    print(f"""
{Colors.CYAN}╭──────────────────────────────────────────────────────────────────────────────╮
│  {Colors.BOLD}{Colors.YELLOW}📚 FLASHCARD #{card_num:<64}{Colors.CYAN}│
│  {Colors.DIM}Topic: {topic[:66]:<66}{Colors.CYAN}│
╰──────────────────────────────────────────────────────────────────────────────╯{Colors.RESET}
""")
    
    # Question
    print(f"  {Colors.BOLD}{Colors.GREEN}❓ QUESTION:{Colors.RESET}")
    wrapped_q = _wrap_text(question, 72)
    for line in wrapped_q:
        print(f"     {line}")
    
    # Answer
    print(f"\n  {Colors.BOLD}{Colors.BLUE}💡 ANSWER:{Colors.RESET}")
    wrapped_a = _wrap_text(answer, 72)
    for line in wrapped_a:
        print(f"     {line}")
    
    # Proofs
    if proofs:
        print(f"\n  {Colors.BOLD}{Colors.DIM}📎 SOURCES ({len(proofs)} reference(s)):{Colors.RESET}")
        for i, proof in enumerate(proofs[:2], 1):
            doc_id = proof.get('doc_id', 'unknown')[:15]
            page = proof.get('page', '?')
            score = proof.get('score', 0)
            text = proof.get('text', '')
            text_preview = " ".join(text.split())[:120]
            if len(text) > 120:
                text_preview += "..."
            print(f"     {Colors.DIM}[{i}] {doc_id} (p.{page}) score: {score:.2f}{Colors.RESET}")
            print(f"         {Colors.DIM}\"{text_preview}\"{Colors.RESET}")

def _wrap_text(text, width):
    """Wrap text to specified width"""
    words = text.split()
    lines = []
    current_line = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 1 <= width:
            current_line.append(word)
            current_length += len(word) + 1
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]
            current_length = len(word)
    
    if current_line:
        lines.append(" ".join(current_line))
    
    return lines if lines else [""]

def print_final_summary(exam_id, num_cards, elapsed_time):
    """Print beautiful final summary"""
    print(f"""
{Colors.GREEN}╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   {Colors.BOLD}✨ DEMO COMPLETED SUCCESSFULLY! ✨{Colors.GREEN}                                       ║
║                                                                              ║
║   {Colors.RESET}{Colors.GREEN}📋 Exam ID:        {exam_id:<55}║
║   📚 Cards Created:  {num_cards:<55}║
║   ⏱️  Time Elapsed:   {elapsed_time:.2f} seconds{' ' * 46}║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝{Colors.RESET}
""")

def print_topics_box(topics):
    """Print topics in a nice box"""
    print(f"\n{Colors.CYAN}  ┌─────────────────────────────────────────────────────────────────────┐{Colors.RESET}")
    print(f"{Colors.CYAN}  │ {Colors.BOLD}{Colors.YELLOW}📂 DISCOVERED TOPICS{Colors.CYAN}                                               │{Colors.RESET}")
    print(f"{Colors.CYAN}  ├─────────────────────────────────────────────────────────────────────┤{Colors.RESET}")
    for t in topics:
        label = t.label[:63] if len(t.label) > 63 else t.label
        print(f"{Colors.CYAN}  │{Colors.RESET}  • {label:<64}{Colors.CYAN}│{Colors.RESET}")
    print(f"{Colors.CYAN}  └─────────────────────────────────────────────────────────────────────┘{Colors.RESET}")

def print_endpoint_result(method: str, endpoint: str, status_code: int, payload: Any):
    """Print endpoint response with consistent formatting."""
    status_color = Colors.GREEN if 200 <= status_code < 300 else Colors.RED
    print(f"\n{Colors.BOLD}{method} {endpoint}{Colors.RESET} -> {status_color}{status_code}{Colors.RESET}")
    try:
        pretty = json.dumps(payload, ensure_ascii=False, indent=2)
    except Exception:
        pretty = str(payload)
    print(pretty)

def _require_ok(resp: httpx.Response, method: str, endpoint: str) -> Dict[str, Any]:
    """Parse JSON and fail fast if request is not successful."""
    try:
        payload: Any = resp.json()
    except Exception:
        payload = {"raw": resp.text}
    print_endpoint_result(method, endpoint, resp.status_code, payload)
    if resp.status_code >= 400:
        raise SystemExit(f"API call failed: {method} {endpoint} -> HTTP {resp.status_code}")
    if isinstance(payload, dict):
        return payload
    return {"data": payload}

def run_api_smoke(args):
    """Run end-to-end API checks through HTTP endpoints."""
    start_time = time.time()
    if not args.api_files:
        raise SystemExit("--api_smoke requires --api_files with one or more document paths")
    for path in args.api_files:
        if not Path(path).exists():
            raise SystemExit(f"File not found: {path}")

    base_url = args.api_base_url.rstrip("/")
    timeout = httpx.Timeout(args.api_timeout)
    print("=== API CLI Smoke Test ===")
    print(f"API base URL: {base_url}")
    print(f"User ID: {args.user_id}")
    print("Running endpoint checks in sequence...")

    with httpx.Client(base_url=base_url, timeout=timeout) as client:
        multipart_files = []
        open_handles = []
        try:
            for file_path in args.api_files:
                p = Path(file_path)
                handle = p.open("rb")
                open_handles.append(handle)
                content_type = mimetypes.guess_type(str(p))[0] or "application/octet-stream"
                multipart_files.append(("files", (p.name, handle, content_type)))

            create_payload = {
                "user_id": args.user_id,
                "title": args.api_exam_title,
                "mode": args.api_exam_mode,
            }
            req_started = time.time()
            try:
                r_create = client.post("/exams/from-upload", data=create_payload, files=multipart_files)
            except httpx.TimeoutException as exc:
                elapsed = time.time() - req_started
                raise SystemExit(
                    f"API call timed out after {elapsed:.1f}s on POST /exams/from-upload. "
                    f"Increase --api_timeout (current: {args.api_timeout})."
                ) from exc
            created = _require_ok(r_create, "POST", "/exams/from-upload")
            exam_id = created.get("exam_id")
            if not exam_id:
                raise SystemExit("Missing exam_id in create response")
        finally:
            for handle in open_handles:
                handle.close()

        r_exams = client.get("/exams", params={"user_id": args.user_id, "limit": 20})
        _require_ok(r_exams, "GET", "/exams")

        r_exam = client.get(f"/exams/{exam_id}", params={"user_id": args.user_id})
        _require_ok(r_exam, "GET", f"/exams/{exam_id}")

        r_topics = client.get(f"/exams/{exam_id}/topics")
        topics_payload = _require_ok(r_topics, "GET", f"/exams/{exam_id}/topics")
        topics = topics_payload.get("topics") or []
        topic_id: Optional[str] = topics[0].get("topic_id") if topics else None

        if topic_id:
            r_generate = client.post(
                f"/exams/{exam_id}/topics/{topic_id}/cards/generate",
                json={"user_id": args.user_id, "difficulty": args.api_difficulty},
            )
            _require_ok(
                r_generate,
                "POST",
                f"/exams/{exam_id}/topics/{topic_id}/cards/generate",
            )
        else:
            print("No topics found, skipping single-card generation step.")

        r_cards = client.get(f"/exams/{exam_id}/cards", params={"limit": 50})
        cards_payload = _require_ok(r_cards, "GET", f"/exams/{exam_id}/cards")
        cards = cards_payload.get("cards") or []
        if not cards:
            raise SystemExit("No cards returned from /cards; cannot continue with review flow")
        card_id = cards[0].get("card_id")
        if not card_id:
            raise SystemExit("First card response is missing card_id")

        r_next = client.get(f"/exams/{exam_id}/session/next-card", params={"user_id": args.user_id})
        _require_ok(r_next, "GET", f"/exams/{exam_id}/session/next-card")

        review_headers = {"Idempotency-Key": f"cli-{exam_id}-{card_id}-{args.api_rating}"}
        r_review = client.post(
            f"/exams/{exam_id}/cards/{card_id}/review",
            data={"user_id": args.user_id, "rating": args.api_rating},
            headers=review_headers,
        )
        _require_ok(r_review, "POST", f"/exams/{exam_id}/cards/{card_id}/review")

        r_progress = client.get(f"/exams/{exam_id}/progress", params={"user_id": args.user_id})
        _require_ok(r_progress, "GET", f"/exams/{exam_id}/progress")

        r_prev = client.get(f"/exams/{exam_id}/session/previous-card", params={"user_id": args.user_id})
        _require_ok(r_prev, "GET", f"/exams/{exam_id}/session/previous-card")

        r_history = client.get(
            f"/exams/{exam_id}/cards/presented-history",
            params={"user_id": args.user_id, "limit": 20},
        )
        _require_ok(r_history, "GET", f"/exams/{exam_id}/cards/presented-history")

    elapsed = time.time() - start_time
    print("\n=== API SMOKE COMPLETED ===")
    print(f"Exam ID: {exam_id}")
    print(f"Cards returned: {len(cards)}")
    print(f"Elapsed: {elapsed:.2f}s")


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Default logging - will be adjusted for demo mode
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)

from app.services.corpus.ingestion import UnsupportedDocumentTypeError, ingest_documents
from app.services.corpus.retrieval import retrieve_with_proofs
from app.data.vector_store import VectorStore
from app.services.corpus.qa import generate_answer
from app.services.exams import create_exam, load_exam, attach_documents, log_event
from app.services.corpus.topics import build_topics_for_exam, list_topics_for_exam
from app.services.generation.routing import answer_in_exam, route_question_to_topic
from app.services.generation.cards import generate_starter_cards

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ingest", nargs="+", help="one or more document paths to ingest")
    p.add_argument("--create_exam", action="store_true", help="create a new exam workspace")
    p.add_argument("--exam_title", default="New Exam", help="title for --create_exam")
    p.add_argument("--exam_mode", default="mastery", help="mastery|exam (for --create_exam)")
    p.add_argument("--user_id", default="local_user", help="local-first user id")
    p.add_argument("--exam_id", help="existing exam id (to attach documents or inspect)")
    p.add_argument("--build_topics", action="store_true", help="build topics for --exam_id")
    p.add_argument("--topic_merge_threshold", type=float, default=0.88, help="merge topics if centroid similarity >= threshold (default 0.88)")
    p.add_argument(
        "--topic_cluster_algorithm",
        default="hdbscan",
        choices=["hdbscan", "auto", "kmeans"],
        help="topic clustering algorithm (default: hdbscan)",
    )
    p.add_argument("--topic_use_umap", action="store_true", help="enable optional UMAP reduction before HDBSCAN on large chunk sets")
    p.add_argument("--topic_umap_n_components", type=int, default=15, help="UMAP components when --topic_use_umap is enabled")
    p.add_argument("--topic_umap_min_chunk_count", type=int, default=300, help="minimum chunk count before applying UMAP")
    p.add_argument("--topic_hdbscan_min_cluster_size", type=int, help="override HDBSCAN min_cluster_size")
    p.add_argument("--topic_hdbscan_min_samples", type=int, help="override HDBSCAN min_samples")
    p.add_argument("--topic_agglomerative_threshold", type=float, default=0.82, help="fallback agglomerative cosine threshold")
    p.add_argument("--topic_id", help="optional: restrict --ask/--answer to this topic_id")
    p.add_argument("--auto_topic", action="store_true", help="auto-route question to topic within --exam_id (for --answer)")
    p.add_argument("--gen_starter_cards", action="store_true", help="generate starter flashcards for --exam_id (requires topics)")
    p.add_argument("--ask", help="question: retrieve proofs only (no LLM answer)")
    p.add_argument("--answer", help="question: retrieve + generate answer with citations")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--min_score", type=float, default=0.4)
    p.add_argument("--demo", nargs="+", help="Full demo: ingest docs, create exam, build topics, generate cards")
    p.add_argument("--quiet", action="store_true", help="Suppress logging output for clean demo")
    p.add_argument("--api_smoke", action="store_true", help="Call HTTP API endpoints end-to-end and print results")
    p.add_argument("--api_base_url", default="http://127.0.0.1:8000", help="FastAPI base URL for --api_smoke")
    p.add_argument("--api_files", nargs="+", help="document paths used by /exams/from-upload in --api_smoke")
    p.add_argument("--api_exam_title", default="CLI API Smoke Exam", help="exam title for --api_smoke")
    p.add_argument("--api_exam_mode", default="mastery", help="mastery|exam for --api_smoke")
    p.add_argument("--api_difficulty", type=int, default=1, help="difficulty for single-card generation in --api_smoke")
    p.add_argument(
        "--api_rating",
        default="almost_knew",
        choices=["i_knew_it", "almost_knew", "learned_now", "dont_understand"],
        help="rating used in review step for --api_smoke",
    )
    p.add_argument("--api_timeout", type=float, default=300.0, help="request timeout seconds for --api_smoke")
    p.add_argument("--out_txt", help="write all CLI print output to this txt file")
    args = p.parse_args()

    original_stdout = sys.stdout
    output_handle = None
    if args.out_txt:
        out_path = Path(args.out_txt)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        output_handle = out_path.open("w", encoding="utf-8")
        sys.stdout = TeeTextOutput(original_stdout, output_handle)
        print(f"Writing CLI output to: {out_path.resolve()}")

    store = VectorStore()

    try:
        _run(args, store)
    finally:
        if output_handle is not None:
            sys.stdout = original_stdout
            output_handle.close()
        # store.db.close();
        pass

def _run(args, store):
    if args.api_smoke:
        run_api_smoke(args)
        return

    # === DEMO MODE: Full automated flow with beautiful output ===
    if args.demo:
        # Suppress all logging for clean presentation
        logging.getLogger().setLevel(logging.CRITICAL)
        for name in ['app.services.corpus.ingestion', 'app.services.generation.graph', 'app.services.llm',
                     'app.services.corpus.topics', 'app.services.generation.cards', 'app.services.corpus.retrieval',
                     'app.data.vector_store', 'app.data.db', 'httpx', 'openai']:
            logging.getLogger(name).setLevel(logging.CRITICAL)
        
        demo_start_time = time.time()
        
        # Clear screen and show banner
        print("\033[2J\033[H", end="")  # Clear screen
        print_banner()
        
        time.sleep(0.5)  # Dramatic pause
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 1: Create Exam Workspace
        # ─────────────────────────────────────────────────────────────────────
        print_step_header(1, 4, "Creating Exam Workspace", "🎯")
        
        exam_id = create_exam(
            store=store,
            user_id=args.user_id,
            title=args.exam_title,
            mode=args.exam_mode,
            info={"created_via": "demo"},
        )
        
        print_success(f"Exam workspace created successfully!")
        print_info(f"Exam ID: {Colors.BOLD}{exam_id}{Colors.RESET}")
        print_info(f"Title: {args.exam_title}")
        print_info(f"Mode: {args.exam_mode}")
        
        time.sleep(0.3)
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 2: Ingest Documents
        # ─────────────────────────────────────────────────────────────────────
        print_step_header(2, 4, "Ingesting Documents", "📄")
        
        print_info("Processing documents...")
        try:
            results = ingest_documents(args.demo, store, user_id=args.user_id, exam_id=exam_id)
        except UnsupportedDocumentTypeError as exc:
            raise SystemExit(str(exc)) from exc
        
        for res in results:
            print_success(f"Ingested: {res.doc_id}")
            print_item(f"Chunks created: {res.num_chunks}", indent=4)
        
        doc_ids = [res.doc_id for res in results]
        attach_documents(store=store, exam_id=exam_id, doc_ids=doc_ids)
        
        print()
        print_success(f"Attached {len(doc_ids)} document(s) to exam")
        
        time.sleep(0.3)
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 3: Build Topics (Clustering)
        # ─────────────────────────────────────────────────────────────────────
        print_step_header(3, 4, "Analyzing & Clustering Content", "🧠")
        
        print_info("Running AI-powered topic extraction...")
        print_info("Clustering document content...")
        
        topics = build_topics_for_exam(
            exam_id=exam_id,
            store=store,
            overwrite=True,
            merge_threshold=args.topic_merge_threshold,
            topic_cluster_algorithm=args.topic_cluster_algorithm,
            use_umap=args.topic_use_umap,
            umap_n_components=args.topic_umap_n_components,
            umap_min_chunk_count=args.topic_umap_min_chunk_count,
            hdbscan_min_cluster_size=args.topic_hdbscan_min_cluster_size,
            hdbscan_min_samples=args.topic_hdbscan_min_samples,
            agglomerative_threshold=args.topic_agglomerative_threshold,
        )
        topic_list = list_topics_for_exam(exam_id=exam_id, store=store)
        
        print_success(f"Identified {len(topic_list)} distinct topic(s)")
        print_topics_box(topic_list)
        
        time.sleep(0.3)
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 4: Generate Flashcards
        # ─────────────────────────────────────────────────────────────────────
        print_step_header(4, 4, "Generating AI Flashcards", "✨")
        
        print_info("Generating intelligent flashcards with RAG...")
        print_info("This may take a moment...")
        print()
        
        card_start_time = time.time()
        cards = generate_starter_cards(
            exam_id=exam_id,
            user_id=args.user_id,
            store=store,
            n=5,
            difficulty=1,
        )
        card_elapsed = time.time() - card_start_time
        
        print_success(f"Generated {len(cards)} flashcard(s)")
        print_info(f"Generation time: {card_elapsed:.2f} seconds")
        if cards:
            print_info(f"Average: {card_elapsed/len(cards):.2f} seconds per card")
        
        # ─────────────────────────────────────────────────────────────────────
        # Display Generated Flashcards
        # ─────────────────────────────────────────────────────────────────────
        print_section_divider()
        
        print(f"""
{Colors.CYAN}╔══════════════════════════════════════════════════════════════════════════════╗
║  {Colors.BOLD}{Colors.YELLOW}📚 GENERATED FLASHCARDS{Colors.CYAN}                                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝{Colors.RESET}
""")
        
        for i, c in enumerate(cards, 1):
            print_card(
                card_num=i,
                topic=c.topic_label,
                question=c.question,
                answer=c.answer,
                proofs=c.proofs
            )
            if i < len(cards):
                print(f"\n{Colors.DIM}{'─' * 80}{Colors.RESET}")
        
        # ─────────────────────────────────────────────────────────────────────
        # Final Summary
        # ─────────────────────────────────────────────────────────────────────
        total_elapsed = time.time() - demo_start_time
        print_final_summary(exam_id, len(cards), total_elapsed)
        
        return

    # ═══════════════════════════════════════════════════════════════════════════════
    # NON-DEMO MODES (Original functionality preserved)
    # ═══════════════════════════════════════════════════════════════════════════════

    if args.gen_starter_cards:
        if not args.exam_id:
            raise SystemExit("--gen_starter_cards requires --exam_id")
        print(f"Generating starter cards for exam_id={args.exam_id}...")
        start_time = time.time()
        cards = generate_starter_cards(
            exam_id=args.exam_id,
            user_id=args.user_id,
            store=store,
            n=5,
            difficulty=1,
        )
        elapsed = time.time() - start_time
        print(f"\nGenerated {len(cards)} starter card(s) in {elapsed:.2f} seconds")
        if cards:
            print(f"Average: {elapsed/len(cards):.2f} seconds per card\n")
        for c in cards:
            print(f"\n{'='*60}")
            print(f"TOPIC: {c.topic_label}")
            print(f"Q: {c.question}")
            print(f"\nA: {c.answer}")
            if c.proofs:
                print(f"\nPROOFS ({len(c.proofs)} sources):")
                for j, proof in enumerate(c.proofs[:3], 1):
                    doc_id = proof.get('doc_id', 'unknown')
                    page = proof.get('page', '?')
                    score = proof.get('score', 0)
                    text = proof.get('text', '')[:150]
                    print(f"  [{j}] doc={doc_id} page={page} score={score:.2f}")
                    print(f"      \"{text}...\"")
        print(f"\n{'='*60}")
        print(f"Total time: {elapsed:.2f} seconds")
        return

    if args.build_topics:
        if not args.exam_id:
            raise SystemExit("--build_topics requires --exam_id")
        topics = build_topics_for_exam(
            exam_id=args.exam_id,
            store=store,
            overwrite=True,
            merge_threshold=args.topic_merge_threshold,
            topic_cluster_algorithm=args.topic_cluster_algorithm,
            use_umap=args.topic_use_umap,
            umap_n_components=args.topic_umap_n_components,
            umap_min_chunk_count=args.topic_umap_min_chunk_count,
            hdbscan_min_cluster_size=args.topic_hdbscan_min_cluster_size,
            hdbscan_min_samples=args.topic_hdbscan_min_samples,
            agglomerative_threshold=args.topic_agglomerative_threshold,
        )
        print(f"Built {len(topics)} topic(s) for exam_id={args.exam_id}")
        for t in list_topics_for_exam(exam_id=args.exam_id, store=store):
            print(f"- {t.topic_id}: {t.label}")
        return

    if args.create_exam:
        eid = create_exam(
            store=store,
            user_id=args.user_id,
            title=args.exam_title,
            mode=args.exam_mode,
            info={"created_via": "cli"},
        )
        print(f"Created exam_id={eid}")
        return

    if args.ingest:
        if not args.exam_id:
            raise SystemExit("--ingest requires --exam_id when using Pinecone backend")
        try:
            results = ingest_documents(args.ingest, store, user_id=args.user_id, exam_id=args.exam_id)
        except UnsupportedDocumentTypeError as exc:
            raise SystemExit(str(exc)) from exc
        for res in results:
            print(f"Ingested doc_id={res.doc_id} chunks={res.num_chunks}")
        doc_ids = [res.doc_id for res in results]
        if args.exam_id and doc_ids:
            attach_documents(store=store, exam_id=args.exam_id, doc_ids=doc_ids)
            log_event(
                store=store,
                user_id=args.user_id,
                exam_id=args.exam_id,
                type="documents_attached",
                payload={"doc_ids": doc_ids},
            )
            ws = load_exam(store=store, exam_id=args.exam_id)
            print(f"Attached documents to exam_id={args.exam_id} (now {len(ws.doc_ids)} doc(s))")

    if args.ask:
        if store.vector_backend == "pinecone":
            if not args.exam_id:
                raise SystemExit("--ask requires --exam_id when using Pinecone backend")
            ex = store.db.get_exam(args.exam_id)
            if ex is None:
                raise SystemExit(f"Exam not found: {args.exam_id}")
            store.set_namespace(f"u:{ex.user_id}|e:{args.exam_id}")
        allowed_chunk_ids = None
        if args.topic_id:
            allowed_chunk_ids = store.db.list_chunk_ids_for_topic(topic_id=args.topic_id)
        proofs = retrieve_with_proofs(
            args.ask,
            k=args.k,
            store=store,
            allowed_chunk_ids=allowed_chunk_ids,
        )
        print(json.dumps([p.model_dump() for p in proofs], ensure_ascii=False, indent=2))

    if args.answer:
        if args.auto_topic:
            if not args.exam_id:
                raise SystemExit("--auto_topic requires --exam_id")
            ans, routes = answer_in_exam(
                exam_id=args.exam_id,
                question=args.answer,
                store=store,
                k=args.k,
                min_score=args.min_score,
                top_n_topics=1,
            )
            if routes:
                r0 = routes[0]
                print(f"\n=== ROUTED TOPIC ===\n{r0.topic_id}: {r0.label} (score={r0.score:.3f})\n")
            print("\n=== ANSWER ===\n")
            print(ans.answer)
            print("\n=== PROOFS ===\n")
            for i, p in enumerate(ans.proofs, 1):
                print(f"S{i} | doc={p.doc_id} page={p.page} score={p.score:.2f}")
                text = " ".join(p.text.split())
                print(f"\"{text}\"")
                print()
            return

        if store.vector_backend == "pinecone":
            if not args.exam_id:
                raise SystemExit("--answer requires --exam_id when using Pinecone backend")
            ex = store.db.get_exam(args.exam_id)
            if ex is None:
                raise SystemExit(f"Exam not found: {args.exam_id}")
            store.set_namespace(f"u:{ex.user_id}|e:{args.exam_id}")
        allowed_chunk_ids = None
        if args.topic_id:
            allowed_chunk_ids = store.db.list_chunk_ids_for_topic(topic_id=args.topic_id)
        ans = generate_answer(
            args.answer,
            k=args.k,
            min_score=args.min_score,
            store=store,
            allowed_chunk_ids=allowed_chunk_ids,
        )
        print("\n=== ANSWER ===\n")
        print(ans.answer)
        print("\n=== PROOFS ===\n")
        for i, p in enumerate(ans.proofs, 1):
            print(f"S{i} | doc={p.doc_id} page={p.page} score={p.score:.2f}")
            text = " ".join(p.text.split())
            print(f"\"{text}\"")
            print()

if __name__ == "__main__":
    main()

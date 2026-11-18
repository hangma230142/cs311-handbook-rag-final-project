#!/usr/bin/env python3
"""
RAG chatbot cho CS Major Handbook (Task 2).
Dùng:
  python run_rag.py --question "How many credits are the major requirements?"
  python run_rag.py --question "..." --model qwen2.5:1.5b --debug
  python run_rag.py --questions questions.csv --out results.csv   (chạy cả bộ câu hỏi)
  python run_rag.py --reindex                                     (xây lại vector store)
"""

import argparse
import bisect
import csv
import logging
import os
import re
import sys
import time
from pathlib import Path

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM, OllamaEmbeddings

logging.getLogger("pypdf").setLevel(logging.ERROR)

# ---------- SETTINGS ----------
PDF_PATH = os.getenv("PDF_PATH", "major_handbook_pdfs/CS_MajorHandbook.pdf")
VECTOR_STORE = os.getenv("VECTOR_STORE", "vector_store_cs")
COLLECTION = "cs_handbook"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:1.5b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
TEMPERATURE = 0.1
NUM_PREDICT = 300        # giới hạn số token trả lời -> chặn vòng lặp vô hạn
REPEAT_PENALTY = 1.15
TOP_K = 5
SCORE_THRESHOLD = 0.52   # trung bình score = 1/(1+distance); dưới mức này coi là lạc đề.
                         # Phải đo lại sau khi đổi chunk (dùng --debug).
NOT_FOUND = "The handbook does not say."

PROMPT_TEMPLATE = """You answer questions about the Computer Science major handbook.
Use ONLY the context below. Do not use any other knowledge.
If the context does not contain the answer, reply exactly: The handbook does not say.
Do not invent course codes, numbers or policies.
Keep the answer short (1-3 sentences).

CONTEXT:
{context}
---
QUESTION: {question}

ANSWER:"""


# ---------- PDF -> TEXT ----------
def clean_text(t: str) -> str:
    """Sửa lỗi ký tự khi trích text từ PDF."""
    for bad, good in {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl"}.items():
        t = t.replace(bad, good)
    t = re.sub(r"(?<=[a-z])8(?=[a-z])", "ti", t)     # Descrip8on -> Description
    t = re.sub(r"(?<=[a-z])'(?=on)", "ti", t)        # Founda'onal -> Foundational
    return re.sub(r"[ \t]+", " ", t).strip()


def load_pdf(path: str) -> Document:
    """Gộp cả handbook thành 1 text, có đánh dấu trang, để chunk không bị cắt cứng ở ranh giới trang."""
    reader = PdfReader(path)
    parts = []
    for i, page in enumerate(reader.pages, start=1):
        parts.append("[Page " + str(i) + "]\n" + clean_text(page.extract_text() or ""))
    return Document(page_content="\n\n".join(parts), metadata={"source": Path(path).name})


COURSE_START = re.compile(r"(?<!\w)(?=[A-Z]{2,5}\d{3}: [A-Z])")   # đầu mô tả môn, vd "CS203: Computer Organization"
# đầu danh sách Major/Minor Requirements và câu "To formally declare ..." -> mỗi cái bắt đầu một chunk mới
HEADING_START = re.compile(r"(?<!\w)(?=(?:Major|Minor) Requirements \(\d+ credits\)(?!\s*\.{3})|To formally declare )")
COURSE_CHUNK_SIZE = 1800                                # đủ to để 1 môn nằm gọn trong 1 chunk


def split_with_pages(doc: Document):
    """Mỗi môn học = 1 chunk (kể cả khi mô tả kéo sang trang sau). Phần còn lại chia theo CHUNK_SIZE."""
    text = doc.page_content
    marks = [(m.start(), int(m.group(1))) for m in re.finditer(r"\[Page (\d+)\]", text)]
    starts = [m[0] for m in marks]
    normal = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""])
    course = RecursiveCharacterTextSplitter(
        chunk_size=COURSE_CHUNK_SIZE, chunk_overlap=0, add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""])

    cuts = sorted({0, len(text)}
                  | {m.start() for m in COURSE_START.finditer(text)}
                  | {m.start() for m in HEADING_START.finditer(text)})
    chunks = []
    for a, b in zip(cuts, cuts[1:]):
        section = text[a:b]
        is_course = re.match(r"\s*[A-Z]{2,5}\d{3}: ", section) is not None
        for c in (course if is_course else normal).create_documents([section], [dict(doc.metadata)]):
            c.metadata["start_index"] += a
            idx = bisect.bisect_right(starts, c.metadata["start_index"]) - 1
            c.metadata["page"] = marks[max(idx, 0)][1]
            chunks.append(c)
    return chunks


def normalize_query(q: str) -> str:
    """'CS 203' -> 'CS203' cho khớp cách handbook viết mã môn."""
    return re.sub(r"\b([A-Z]{2,5}) (\d{3})\b", r"\1\2", q)


# ---------- VECTOR STORE ----------
def get_db(embeddings, reindex=False) -> Chroma:
    store = Path(VECTOR_STORE)
    has_data = store.exists() and any(store.iterdir())   # thư mục rỗng = chưa có index
    if has_data and not reindex:
        return Chroma(collection_name=COLLECTION, embedding_function=embeddings,
                      persist_directory=VECTOR_STORE)
    if has_data:
        import shutil
        for child in store.iterdir():                    # xóa nội dung, giữ thư mục (có thể là volume mount)
            shutil.rmtree(child) if child.is_dir() else child.unlink()
    if not Path(PDF_PATH).exists():
        print("Không thấy file: " + PDF_PATH)
        sys.exit(1)
    chunks = split_with_pages(load_pdf(PDF_PATH))
    print("Indexing " + str(len(chunks)) + " chunks...")
    return Chroma.from_documents(chunks, embeddings, collection_name=COLLECTION,
                                 persist_directory=VECTOR_STORE)


# ---------- ASK ----------
def ask(question: str, db: Chroma, llm: OllamaLLM) -> dict:
    question = normalize_query(question)
    results = db.similarity_search_with_score(question, k=TOP_K)
    sources = [{"page": d.metadata.get("page"), "score": round(1 / (1 + s), 4),
                "content": d.page_content} for d, s in results]
    avg = sum(s["score"] for s in sources) / len(sources) if sources else 0
    if avg < SCORE_THRESHOLD:
        # Lạc đề: không gọi LLM, trả lời thẳng.
        return {"answer": NOT_FOUND, "sources": sources, "avg_score": avg, "fallback": True}
    context = "\n\n---\n\n".join("(page " + str(s["page"]) + ")\n" + s["content"] for s in sources)
    prompt = PROMPT_TEMPLATE.replace("{context}", context).replace("{question}", question)
    answer = llm.invoke(prompt).strip()
    return {"answer": answer, "sources": sources, "avg_score": avg, "fallback": False}


def make_llm(model: str) -> OllamaLLM:
    return OllamaLLM(model=model, base_url=OLLAMA_URL, temperature=TEMPERATURE,
                     num_predict=NUM_PREDICT, repeat_penalty=REPEAT_PENALTY)


def load_questions(path: str):
    """Đọc .txt (mỗi dòng 'category | question', dòng # bị bỏ qua) hoặc .csv (cột 'question')."""
    if path.endswith(".csv"):
        with open(path, newline="", encoding="utf-8") as f:
            return [(r.get("category", ""), r["question"]) for r in csv.DictReader(f)]
    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        cat, _, q = line.partition("|")
        rows.append((cat.strip(), q.strip()) if q else ("", line))
    return rows


def run_batch(path: str, out: str, db, llm, model: str):
    rows = load_questions(path)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "category", "question", "answer", "retrieved_sources",
                    "used_fallback", "seconds", "avg_score", "model"])
        for i, (cat, q) in enumerate(rows, start=1):
            t0 = time.time()
            r = ask(q, db, llm)
            sec = round(time.time() - t0, 1)
            pages = ",".join(str(s["page"]) for s in r["sources"])
            w.writerow([i, cat, q, r["answer"], pages, r["fallback"], sec,
                        round(r["avg_score"], 4), model])
            f.flush()
            print("[" + str(i) + "/" + str(len(rows)) + "] " + q + "\n   done in " + str(sec) + "s")
    print("Saved results to " + out)


def main():
    p = argparse.ArgumentParser(description="CS Handbook RAG chatbot")
    p.add_argument("--question", "-q", default="How many credits are the major requirements?")
    p.add_argument("--questions", help="file câu hỏi .txt hoặc .csv (chạy cả bộ)")
    p.add_argument("--out", default="results.csv")
    p.add_argument("--model", default=LLM_MODEL)
    p.add_argument("--chat", action="store_true", help="hỏi nhiều câu liên tiếp (gõ exit để thoát)")
    p.add_argument("--reindex", action="store_true")
    p.add_argument("--debug", action="store_true", help="in score và các chunk lấy được")
    args = p.parse_args()

    db = get_db(OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL), args.reindex)
    llm = make_llm(args.model)

    if args.questions:
        run_batch(args.questions, args.out, db, llm, args.model)
        return

    if args.chat:
        print("Chat với CS Handbook (model: " + args.model + "). Gõ 'exit' để thoát.")
        while True:
            try:
                q = input("\nQ: ").strip()
            except EOFError:
                break
            if q.lower() in ("exit", "quit", ""):
                break
            r = ask(q, db, llm)
            print("A: " + r["answer"])
            print("   (" + ", ".join("page " + str(x["page"]) for x in r["sources"]) + ")")
        return

    print("Q: " + args.question)
    r = ask(args.question, db, llm)
    print("\nA: " + r["answer"] + "\n")
    if r["fallback"]:
        print("(score thấp: " + str(round(r["avg_score"], 4)) + " < " + str(SCORE_THRESHOLD) + ")")
    print("Sources: " + ", ".join("page " + str(s["page"]) + " (score=" + str(s["score"]) + ")"
                                  for s in r["sources"]))
    if args.debug:
        print("avg_score=" + str(round(r["avg_score"], 4)))
        for s in r["sources"]:
            print("\n--- page " + str(s["page"]) + " ---\n" + s["content"])


if __name__ == "__main__":
    main()

"""RAG：講義索引（依課程分開）+ 教授優先答案 + Gemini 回答。"""
import hashlib
import os
import tempfile

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
import db

GENERAL_MARKER = "[GENERAL]"
GENERAL_NOTE = {
    "zh": "*（提醒：這個概念在目前的講義中沒有明確說明，以下是一般性的解釋。）*",
    "en": "*(Note: This concept is not explicitly detailed in your current handout chunks, but generally speaking...)*",
}


# ---------------- components ----------------
def get_embeddings():
    if config.EMBEDDING_PROVIDER == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=config.HF_EMBEDDING_MODEL,
                                     encode_kwargs={"normalize_embeddings": True})
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    return GoogleGenerativeAIEmbeddings(model=config.GOOGLE_EMBEDDING_MODEL,
                                        google_api_key=config.GOOGLE_API_KEY)


def get_vectorstore():
    # 不同 embedding 的向量維度不同，collection 分開放，切換時不會互相污染
    return Chroma(collection_name=f"handouts_{config.EMBEDDING_PROVIDER}",
                  persist_directory=config.CHROMA_DIR,
                  embedding_function=get_embeddings())


def get_llm():
    if not config.GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY 未設定")
    return ChatGoogleGenerativeAI(model=config.GEMINI_MODEL, api_key=config.GOOGLE_API_KEY,
                                  temperature=config.LLM_TEMPERATURE)


# ---------------- handout management ----------------
def _course_file_filter(course, file_hash):
    return {"$and": [{"course": course}, {"file_hash": file_hash}]}


def delete_handout(vs, course, file_hash):
    existing = vs.get(where=_course_file_filter(course, file_hash))
    if existing["ids"]:
        vs.delete(ids=existing["ids"])
    return len(existing["ids"])


def index_pdf(vs, file_bytes: bytes, filename: str, course: str) -> int:
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    delete_handout(vs, course, file_hash)  # 重複上傳 = 覆蓋，不會產生重複片段

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        pages = PyPDFLoader(tmp_path).load()
    finally:
        os.remove(tmp_path)

    pages = [p for p in pages if p.page_content.strip()]
    if not pages:
        raise ValueError("這份 PDF 抽不出文字（可能是掃描圖片），請先做 OCR 再上傳。")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    prefix = hashlib.sha1(f"{course}|{file_hash}".encode()).hexdigest()[:16]
    for i, ch in enumerate(chunks):
        page = int(ch.metadata.get("page", 0)) + 1
        ch.metadata = {"course": course, "source": filename, "file_hash": file_hash,
                       "page": page, "chunk": i}
    vs.add_documents(chunks, ids=[f"{prefix}-{i}" for i in range(len(chunks))])
    return len(chunks)


def list_handouts(vs, course):
    res = vs.get(where={"course": course}, include=["metadatas"])
    files = {}
    for m in res["metadatas"]:
        f = files.setdefault(m["file_hash"], {"file_hash": m["file_hash"], "source": m["source"],
                                              "chunks": 0, "pages": 0})
        f["chunks"] += 1
        f["pages"] = max(f["pages"], m.get("page", 0))
    return sorted(files.values(), key=lambda f: f["source"])


# ---------------- question answering ----------------
def retrieve(vs, query, course):
    courses = [course] if course == config.GENERAL_COURSE else [course, config.GENERAL_COURSE]
    flt = {"course": courses[0]} if len(courses) == 1 else {"course": {"$in": courses}}
    return vs.similarity_search(query, k=config.RETRIEVER_K, filter=flt)


def _is_chinese(text):
    return any("一" <= ch <= "鿿" for ch in text)


def _to_text(content):
    if isinstance(content, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content)


def build_prompt(query, course, docs, history):
    if docs:
        context = "\n\n".join(
            f"[{i}] ({d.metadata.get('source', '?')}, p.{d.metadata.get('page', '?')})\n{d.page_content}"
            for i, d in enumerate(docs, 1))
    else:
        context = "(No handout excerpts were found for this course.)"
    convo = "\n".join(f"{m['role']}: {m['content'][:500]}" for m in (history or [])[-6:]) or "(none)"
    return f"""You are an AI study assistant for the university course "{course}".
Answer the student's question using the handout excerpts below.

Rules:
- Reply in the same language the student uses (Traditional Chinese if the student writes Chinese).
- When you use an excerpt, cite it inline as [1], [2], ...
- For summary / review requests, synthesize across the excerpts.
- If the excerpts do not contain enough information, you may use general academic knowledge,
  but then the FIRST line of your reply must be exactly: {GENERAL_MARKER}
- Never invent handout content, page numbers, deadlines, or exam information.

Recent conversation:
{convo}

Handout excerpts:
{context}

Student question: {query}"""


def _serialize(docs):
    return [{"source": d.metadata.get("source", "?"), "page": d.metadata.get("page", "?"),
             "text": d.page_content} for d in docs]


def answer_question(llm, vs, query, course, history=None):
    verified = db.find_verified_answer(course, query)
    if verified:
        return {"answer": f"⭐ **[Professor Verified Answer]**\n\n{verified}", "docs": [], "source": "professor"}

    try:
        docs = retrieve(vs, query, course)
    except Exception as e:
        print(f"[rag] retrieval failed: {e!r}")
        docs = []

    try:
        text = _to_text(llm.invoke(build_prompt(query, course, docs, history)).content).strip()
    except Exception as e:
        print(f"[rag] LLM failed: {e!r}")
        return {"answer": f"⚠️ AI 服務暫時無法回應，請稍後再試。（{type(e).__name__}）",
                "docs": _serialize(docs), "source": "error"}

    source = "handout"
    if text.startswith(GENERAL_MARKER):
        source = "general"
        note = GENERAL_NOTE["zh" if _is_chinese(query) else "en"]
        text = note + "\n\n" + text[len(GENERAL_MARKER):].lstrip()
    return {"answer": text, "docs": _serialize(docs), "source": source}

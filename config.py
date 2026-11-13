"""集中管理所有設定：.env → 環境變數 → Streamlit Secrets，依序讀取。"""
import os
import secrets as _secrets

from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))


def _get(key: str, default=None):
    value = os.getenv(key)
    if value not in (None, ""):
        return value
    try:
        import streamlit as st
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return default


# ---- LLM ----
GOOGLE_API_KEY = _get("GOOGLE_API_KEY")
GEMINI_MODEL = _get("GEMINI_MODEL", "gemini-3.6-flash")
LLM_TEMPERATURE = float(_get("LLM_TEMPERATURE", "0.2"))

# ---- Embeddings ----
# google：用 Gemini embedding API，不需安裝 torch，部署快、支援中文（預設）
# huggingface：本機模型，需 requirements-local.txt
EMBEDDING_PROVIDER = _get("EMBEDDING_PROVIDER", "google").lower()
GOOGLE_EMBEDDING_MODEL = _get("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001")
HF_EMBEDDING_MODEL = _get("HF_EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# ---- RAG ----
CHUNK_SIZE = int(_get("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(_get("CHUNK_OVERLAP", "120"))
RETRIEVER_K = int(_get("RETRIEVER_K", "4"))
MAX_QUERY_CHARS = int(_get("MAX_QUERY_CHARS", "2000"))

# ---- Courses ----
GENERAL_COURSE = "General Handouts"
COURSES = [c.strip() for c in _get(
    "COURSES", "General Handouts,Operating Systems,Computer Architecture,Data Structures"
).split(",") if c.strip()]
if GENERAL_COURSE not in COURSES:
    COURSES.insert(0, GENERAL_COURSE)
ALL_COURSES = "All Courses"

# ---- Storage：所有會變動的資料都在 DATA_DIR，部署時只要掛一個磁碟就好 ----
DATA_DIR = _get("DATA_DIR", os.path.join(BASE_DIR, "data"))
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "portal.db")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")

# ---- Auth / privacy ----
DEMO_MODE = _get("DEMO_MODE", "false").lower() == "true"
DEMO_PASSWORD = _get("DEMO_PASSWORD")


def _load_app_secret() -> str:
    """匿名身分雜湊用的金鑰。沒設定就自動產生並存在 DATA_DIR（重啟後不變）。"""
    env_secret = _get("APP_SECRET")
    if env_secret:
        return env_secret
    path = os.path.join(DATA_DIR, ".app_secret")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    value = _secrets.token_hex(32)
    with open(path, "w", encoding="utf-8") as f:
        f.write(value)
    return value


APP_SECRET = _load_app_secret()

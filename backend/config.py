
"""Central configuration for the Customer Support Bot."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Project layout
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent

# Load local environment files
load_dotenv(PROJECT_DIR / ".env")
load_dotenv(BACKEND_DIR / ".env")


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, "").strip() or default


# Frontend
FRONTEND_DIR = BACKEND_DIR / "frontend"
if not FRONTEND_DIR.is_dir():
    FRONTEND_DIR = PROJECT_DIR / "frontend"

# Qdrant Cloud
QDRANT_URL = _env("QDRANT_URL")
QDRANT_API_KEY = _env("QDRANT_API_KEY")

# Secrets
GEMINI_API_KEY: str = _env("GEMINI_API_KEY")
ADMIN_PASSWORD: str = _env("ADMIN_PASSWORD")

# Models
GEN_MODEL: str = _env("GEN_MODEL", "gemini-3.8-flash")
EMBED_MODEL: str = _env("EMBED_MODEL", "gemini-embedding-001")
GENERATION_TEMPERATURE: float = 0.2

# Gemini settings
GEN_THINKING_BUDGET: int = 0
MAX_OUTPUT_TOKENS: int = 400

# RAG settings
CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 50
TOP_K: int = 4
MIN_SIMILARITY: float = 0.35
MAX_HISTORY_TURNS: int = 4
EMBED_BATCH_SIZE: int = 50
MAX_RETRIES: int = 3
COLLECTION_NAME: str = "support_docs"

# Input limits
MAX_MESSAGE_CHARS: int = 1000
MAX_FILE_BYTES: int = 10 * 1024 * 1024
SUPPORTED_EXTENSIONS: tuple[str, ...] = (".pdf", ".txt", ".docx")

# Server
PORT: int = int(_env("PORT", "8000"))

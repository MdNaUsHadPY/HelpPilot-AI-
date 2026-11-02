"""Central configuration for the Customer Support Bot.

Every tunable setting lives here. Secrets and model names can be overridden
with environment variables (Replit Secrets, or a local .env file).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# --- Project layout (anchored to this file, so it works from any directory) ---
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent

# Use the bundled frontend on Vercel, or the original folder locally
FRONTEND_DIR = BACKEND_DIR / "frontend"
if not FRONTEND_DIR.is_dir():
    FRONTEND_DIR = PROJECT_DIR / "frontend"
CHROMA_DIR = PROJECT_DIR / "chroma_db"

# Load a local .env if one exists. On Replit the Secrets are already
# environment variables, and real environment variables always win.
load_dotenv(PROJECT_DIR / ".env")
load_dotenv(BACKEND_DIR / ".env")


def _env(name: str, default: str = "") -> str:
    """Read an environment variable, treating empty values as 'not set'."""
    return os.getenv(name, "").strip() or default


# --- Secrets (never print or return these) ---
GEMINI_API_KEY: str = _env("GEMINI_API_KEY")
ADMIN_PASSWORD: str = _env("ADMIN_PASSWORD")  # optional; empty = admin is open

# --- Models (change these if Google renames or retires a model) ---
GEN_MODEL: str = _env("GEN_MODEL", "gemini-3.8-flash")
EMBED_MODEL: str = _env("EMBED_MODEL", "gemini-embedding-001")
GENERATION_TEMPERATURE: float = 0.2

# Gemini response optimization
GEN_THINKING_BUDGET: int = 0
MAX_OUTPUT_TOKENS: int = 400
# --- RAG settings ---
CHUNK_SIZE: int = 500          # target characters per chunk
CHUNK_OVERLAP: int = 50        # characters shared between neighbouring chunks
TOP_K: int = 4                 # chunks retrieved per question
MIN_SIMILARITY: float = 0.35   # cosine similarity below this is ignored
MAX_HISTORY_TURNS: int = 4     # most recent chat messages sent to the model
EMBED_BATCH_SIZE: int = 50     # chunks embedded per API call
MAX_RETRIES: int = 3           # attempts for rate-limit / server errors
COLLECTION_NAME: str = "support_docs"

# --- Input limits ---
MAX_MESSAGE_CHARS: int = 1000
MAX_FILE_BYTES: int = 10 * 1024 * 1024  # 10 MB
SUPPORTED_EXTENSIONS: tuple[str, ...] = (".pdf", ".txt", ".docx")

# --- Server ---
PORT: int = int(_env("PORT", "8000"))

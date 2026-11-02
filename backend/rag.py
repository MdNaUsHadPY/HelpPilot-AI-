"""RAG pipeline: extract text, chunk, embed, store in ChromaDB, retrieve.

Flow: ingest_document() -> extract_text() -> chunk_text() -> embed_texts()
      -> ChromaDB, and later retrieve() finds the chunks closest to a question.
"""

import io
import logging
import re
import time
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, TypeVar

import uuid

from qdrant_client import QdrantClient, models

from docx import Document
from google import genai
from google.genai import errors, types
from pypdf import PdfReader

import config

logger = logging.getLogger("support_bot.rag")

T = TypeVar("T")

# HTTP codes worth retrying: rate limit and temporary server trouble.
RETRYABLE_CODES = {429, 500, 502, 503, 504}
UPSERT_BATCH_SIZE = 500


class BotError(Exception):
    """An error whose message is safe and friendly to show to the user."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned by retrieve(), with its similarity score (0 to 1)."""

    text: str
    source: str
    score: float


# --------------------------------------------------------------------------
# Gemini client and retry
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_client() -> genai.Client:
    """Create the shared Gemini client (reads the key from the environment)."""
    if not config.GEMINI_API_KEY:
        raise BotError(
            "The server is missing its GEMINI_API_KEY. Add it to the Secrets "
            "(Replit) or the .env file (local) and restart the app.",
            status_code=503,
        )
    return genai.Client(api_key=config.GEMINI_API_KEY)


def _friendly_api_error(exc: errors.APIError) -> BotError:
    """Translate a Gemini API error into a message a user can act on."""
    code = exc.code
    text = str(exc.message or "").lower()
    if code == 404:
        return BotError(
            f"The Gemini model was not found. Please change GEN_MODEL "
            f"('{config.GEN_MODEL}') or EMBED_MODEL ('{config.EMBED_MODEL}') "
            "to a model name that is available to your API key.",
            status_code=502,
        )
    if code in (401, 403) or (code == 400 and "api key" in text):
        return BotError(
            "The Gemini API key was rejected. Check that GEMINI_API_KEY is "
            "correct and has access to the Gemini API.",
            status_code=502,
        )
    if code == 429:
        return BotError(
            "The AI service is busy or the free quota has been reached. "
            "Please wait a minute and try again.",
            status_code=503,
        )
    if code in RETRYABLE_CODES:
        return BotError(
            "The AI service is temporarily unavailable. Please try again shortly.",
            status_code=503,
        )
    logger.error("Unexpected Gemini API error %s: %s", code, exc.message)
    return BotError("The AI service returned an error. Please try again.", 502)


def call_with_retry(call: Callable[[], T]) -> T:
    """Run a Gemini call, retrying rate-limit and server errors with backoff."""
    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            return call()
        except errors.APIError as exc:
            if exc.code in RETRYABLE_CODES and attempt < config.MAX_RETRIES:
                delay = 2 ** (attempt - 1)  # 1s, 2s, 4s, 8s
                logger.warning(
                    "Gemini error %s (attempt %d/%d); retrying in %ds",
                    exc.code, attempt, config.MAX_RETRIES, delay,
                )
                time.sleep(delay)
                continue
            raise _friendly_api_error(exc) from exc
    raise BotError("The AI service did not respond. Please try again.", 503)


# --------------------------------------------------------------------------
# 1. Text extraction
# --------------------------------------------------------------------------

def _read_pdf(data: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise BotError("This PDF is password-protected and cannot be read.")
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    except BotError:
        raise
    except Exception as exc:  # pypdf raises many different parser errors
        raise BotError("This PDF could not be read. It may be corrupted.") from exc


def _read_docx(data: bytes) -> str:
    try:
        document = Document(io.BytesIO(data))
    except Exception as exc:  # invalid zip / XML
        raise BotError("This DOCX file could not be read. It may be corrupted.") from exc
    parts = [p.text for p in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text.strip() for cell in row.cells))
    return "\n\n".join(parts)


def _read_txt(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def extract_text(filename: str, data: bytes) -> str:
    """Return the plain text of a PDF, TXT or DOCX file, or raise BotError."""
    extension = Path(filename).suffix.lower()
    if extension not in config.SUPPORTED_EXTENSIONS:
        raise BotError(
            f"Unsupported file type '{extension or filename}'. "
            "Please upload a PDF, TXT or DOCX file."
        )
    if not data:
        raise BotError("The file is empty.")

    readers = {".pdf": _read_pdf, ".docx": _read_docx, ".txt": _read_txt}
    text = readers[extension](data)
    if not text.strip():
        raise BotError(
            "No readable text was found in this file. "
            "Scanned PDFs (images only) are not supported."
        )
    return text


# --------------------------------------------------------------------------
# 2. Chunking
# --------------------------------------------------------------------------

def _split_long_unit(unit: str, size: int) -> list[str]:
    """Break a unit that is longer than `size` at word boundaries."""
    pieces: list[str] = []
    current = ""
    for word in unit.split(" "):
        if current and len(current) + 1 + len(word) > size:
            pieces.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        pieces.append(current)
    return pieces


def _split_into_units(text: str, size: int) -> list[str]:
    """Split text into sentences, never crossing a paragraph boundary."""
    units: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = re.sub(r"\s+", " ", paragraph).strip()
        if not paragraph:
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
            if len(sentence) > size:
                units.extend(_split_long_unit(sentence, size))
            else:
                units.append(sentence)
    return units


def _overlap_tail(chunk: str, overlap: int) -> str:
    """Last ~`overlap` characters of a chunk, starting at a word boundary."""
    if overlap <= 0 or len(chunk) <= overlap:
        return ""
    tail = chunk[-overlap:]
    first_space = tail.find(" ")
    return tail[first_space + 1:] if first_space != -1 else ""


def chunk_text(
    text: str, size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP
) -> list[str]:
    """Split text into chunks of at most `size` characters.

    Chunks end at sentence boundaries where possible, and each chunk starts
    with a short overlap from the previous one so context is not lost.
    """
    chunks: list[str] = []
    current = ""
    for unit in _split_into_units(text, size):
        if current and len(current) + 1 + len(unit) > size:
            chunks.append(current)
            tail = _overlap_tail(current, overlap)
            current = f"{tail} {unit}" if tail and len(tail) + 1 + len(unit) <= size else unit
        else:
            current = f"{current} {unit}".strip()
    if current:
        chunks.append(current)
    return chunks


# --------------------------------------------------------------------------
# 3. Embedding
# --------------------------------------------------------------------------

def embed_texts(texts: list[str], task_type: str) -> list[list[float]]:
    """Embed texts in batches. task_type is RETRIEVAL_DOCUMENT or RETRIEVAL_QUERY."""
    client = get_client()
    embed_config = types.EmbedContentConfig(task_type=task_type)
    vectors: list[list[float]] = []
    for start in range(0, len(texts), config.EMBED_BATCH_SIZE):
        batch = texts[start:start + config.EMBED_BATCH_SIZE]
        response = call_with_retry(
            lambda batch=batch: client.models.embed_content(
                model=config.EMBED_MODEL, contents=batch, config=embed_config
            )
        )
        embeddings = response.embeddings or []
        if len(embeddings) != len(batch):
            raise BotError("The embedding service returned an unexpected response.", 502)
        vectors.extend(list(item.values) for item in embeddings)
    return vectors


# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# 4. Storage - Qdrant Cloud
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    """Connect to the hosted Qdrant database."""
    if not config.QDRANT_URL or not config.QDRANT_API_KEY:
        raise BotError(
            "The server is missing QDRANT_URL or QDRANT_API_KEY.",
            status_code=503,
        )

    return QdrantClient(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
        timeout=30,
    )


def ensure_collection(vector_size: int) -> None:
    """Create the Qdrant collection if it does not exist."""
    client = get_qdrant_client()

    if not client.collection_exists(config.COLLECTION_NAME):
        client.create_collection(
            collection_name=config.COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )


def _point_id(filename: str, index: int) -> str:
    """Generate a stable UUID for each document chunk."""
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"helppilot:{filename}:{index}",
        )
    )


def ingest_document(filename: str, data: bytes) -> int:
    """Extract, embed and store a document in Qdrant."""
    name = Path(filename).name
    chunks = chunk_text(extract_text(name, data))

    if not chunks:
        raise BotError("No readable text was found in this file.")

    # Generate embeddings before replacing the existing document.
    vectors = embed_texts(chunks, "RETRIEVAL_DOCUMENT")

    ensure_collection(len(vectors[0]))
    client = get_qdrant_client()

    # Replace any previous version of this document.
    delete_document(name)

    points = [
        models.PointStruct(
            id=_point_id(name, i),
            vector=vectors[i],
            payload={
                "source": name,
                "chunk_index": i,
                "text": chunks[i],
            },
        )
        for i in range(len(chunks))
    ]

    for start in range(0, len(points), UPSERT_BATCH_SIZE):
        client.upsert(
            collection_name=config.COLLECTION_NAME,
            points=points[start:start + UPSERT_BATCH_SIZE],
            wait=True,
        )

    return len(chunks)


# --------------------------------------------------------------------------
# 5. Retrieval
# --------------------------------------------------------------------------

def retrieve(question: str) -> list[RetrievedChunk]:
    """Return the most relevant document chunks."""
    client = get_qdrant_client()

    if not client.collection_exists(config.COLLECTION_NAME):
        return []

    total = client.count(
        collection_name=config.COLLECTION_NAME,
        exact=True,
    ).count

    if total == 0:
        return []

    query_vector = embed_texts(
        [question],
        "RETRIEVAL_QUERY",
    )[0]

    result = client.query_points(
        collection_name=config.COLLECTION_NAME,
        query=query_vector,
        limit=min(config.TOP_K, total),
        with_payload=True,
    )

    chunks: list[RetrievedChunk] = []

    for point in result.points:
        payload = point.payload or {}
        similarity = float(point.score or 0.0)

        source = str(payload.get("source", "Unknown"))
        text = str(payload.get("text", ""))

        logger.info(
            "Chunk from %s scored %.3f",
            source,
            similarity,
        )

        if text and similarity >= config.MIN_SIMILARITY:
            chunks.append(
                RetrievedChunk(
                    text=text,
                    source=source,
                    score=round(similarity, 4),
                )
            )

    return chunks


# --------------------------------------------------------------------------
# 6. Document management
# --------------------------------------------------------------------------

def list_documents() -> list[dict[str, str | int]]:
    """Return indexed documents and their chunk counts."""
    client = get_qdrant_client()

    if not client.collection_exists(config.COLLECTION_NAME):
        return []

    counts: Counter[str] = Counter()
    offset = None

    while True:
        points, offset = client.scroll(
            collection_name=config.COLLECTION_NAME,
            limit=500,
            offset=offset,
            with_payload=["source"],
            with_vectors=False,
        )

        for point in points:
            payload = point.payload or {}
            source = payload.get("source")

            if source:
                counts[str(source)] += 1

        if offset is None:
            break

    return [
        {"filename": name, "chunks": count}
        for name, count in sorted(counts.items())
    ]


def delete_document(filename: str) -> int:
    """Delete all chunks belonging to a document."""
    client = get_qdrant_client()

    if not client.collection_exists(config.COLLECTION_NAME):
        return 0

    result = client.count(
        collection_name=config.COLLECTION_NAME,
        count_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="source",
                    match=models.MatchValue(value=filename),
                )
            ]
        ),
        exact=True,
    )

    if result.count:
        client.delete(
            collection_name=config.COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source",
                            match=models.MatchValue(value=filename),
                        )
                    ]
                )
            ),
            wait=True,
        )

    return result.count

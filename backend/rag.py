
"""RAG pipeline: extract text, chunk, embed, store in Qdrant, and retrieve.

Flow:
ingest_document() -> extract_text() -> chunk_text() -> embed_texts()
-> Qdrant Cloud.

retrieve() finds the document chunks closest to a question.
"""

import io
import logging
import re
import time
import uuid

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, TypeVar

from docx import Document
from google import genai
from google.genai import errors, types
from pypdf import PdfReader
from qdrant_client import QdrantClient, models

import config


logger = logging.getLogger("support_bot.rag")

T = TypeVar("T")

RETRYABLE_CODES = {429, 500, 502, 503, 504}
UPSERT_BATCH_SIZE = 500


# --------------------------------------------------------------------------
# Custom errors and data classes
# --------------------------------------------------------------------------

class BotError(Exception):
    """An error whose message is safe to show to the user."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@dataclass(frozen=True)
class RetrievedChunk:
    """A retrieved document chunk with its similarity score."""

    text: str
    source: str
    score: float


# --------------------------------------------------------------------------
# Gemini client and retry
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_client() -> genai.Client:
    """Create the shared Gemini client."""

    if not config.GEMINI_API_KEY:
        raise BotError(
            "The server is missing its GEMINI_API_KEY.",
            status_code=503,
        )

    return genai.Client(
        api_key=config.GEMINI_API_KEY,
        http_options=types.HttpOptions(
            timeout=45000,
            retry_options=types.HttpRetryOptions(
                attempts=1,
            ),
        ),
    )


def _friendly_api_error(exc: errors.APIError) -> BotError:
    """Convert Gemini API errors into safe user-facing messages."""

    code = exc.code

    if code in (400, 401, 403):
        logger.error(
            "Gemini authentication or request error: code=%s, message=%s",
            code,
            exc.message,
        )
        return BotError(
            "The Gemini API key was rejected or the request was invalid. "
            "Check GEMINI_API_KEY and your API access.",
            status_code=502,
        )

    if code == 429:
        return BotError(
            "The AI service is busy or the free quota has been reached. "
            "Please wait a minute and try again.",
            status_code=503,
        )

    if code in RETRYABLE_CODES:
        logger.error(
            "Gemini final API failure: code=%s, message=%s",
            code,
            exc.message,
        )
        return BotError(
            "The AI service is temporarily unavailable. "
            "Please try again shortly.",
            status_code=503,
        )

    logger.error(
        "Unexpected Gemini API error %s: %s",
        code,
        exc.message,
    )

    return BotError(
        "The AI service returned an error. Please try again.",
        status_code=502,
    )


def call_with_retry(call: Callable[[], T]) -> T:
    """Run a Gemini call with controlled retries."""

    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            return call()

        except errors.APIError as exc:
            if (
                exc.code in RETRYABLE_CODES
                and attempt < config.MAX_RETRIES
            ):
                delay = 2 ** (attempt - 1)

                logger.warning(
                    "Gemini error %s (attempt %d/%d); retrying in %ds",
                    exc.code,
                    attempt,
                    config.MAX_RETRIES,
                    delay,
                )

                time.sleep(delay)
                continue

            raise _friendly_api_error(exc) from exc

    raise BotError(
        "The AI service did not respond. Please try again.",
        status_code=503,
    )


# --------------------------------------------------------------------------
# Qdrant client
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    """Create the shared Qdrant Cloud client."""

    if not config.QDRANT_URL or not config.QDRANT_API_KEY:
        raise BotError(
            "The server is missing its Qdrant configuration.",
            status_code=503,
        )

    return QdrantClient(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
        timeout=30,
    )


def ensure_collection(vector_size: int) -> None:
    """Create the Qdrant collection and required payload index."""

    client = get_qdrant_client()

    if not client.collection_exists(config.COLLECTION_NAME):
        client.create_collection(
            collection_name=config.COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    client.create_payload_index(
        collection_name=config.COLLECTION_NAME,
        field_name="source",
        field_schema=models.PayloadSchemaType.KEYWORD,
        wait=True,
    )


# --------------------------------------------------------------------------
# 1. Text extraction
# --------------------------------------------------------------------------

def _read_pdf(data: bytes) -> str:
    """Extract text from a PDF file."""

    try:
        reader = PdfReader(io.BytesIO(data))

        if reader.is_encrypted:
            raise BotError(
                "This PDF is password-protected and cannot be read."
            )

        return "\n\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )

    except BotError:
        raise

    except Exception as exc:
        raise BotError(
            "This PDF could not be read. It may be corrupted."
        ) from exc


def _read_docx(data: bytes) -> str:
    """Extract text from a DOCX file, including tables."""

    try:
        document = Document(io.BytesIO(data))

    except Exception as exc:
        raise BotError(
            "This DOCX file could not be read. It may be corrupted."
        ) from exc

    parts = [
        paragraph.text
        for paragraph in document.paragraphs
    ]

    for table in document.tables:
        for row in table.rows:
            parts.append(
                " | ".join(
                    cell.text.strip()
                    for cell in row.cells
                )
            )

    return "\n\n".join(parts)


def _read_txt(data: bytes) -> str:
    """Read a text file."""

    try:
        return data.decode("utf-8-sig")

    except UnicodeDecodeError:
        return data.decode("latin-1")


def extract_text(filename: str, data: bytes) -> str:
    """Extract text from PDF, TXT, or DOCX files."""

    extension = Path(filename).suffix.lower()

    if extension not in config.SUPPORTED_EXTENSIONS:
        raise BotError(
            f"Unsupported file type '{extension or filename}'. "
            "Please upload a PDF, TXT or DOCX file."
        )

    if not data:
        raise BotError("The file is empty.")

    readers = {
        ".pdf": _read_pdf,
        ".docx": _read_docx,
        ".txt": _read_txt,
    }

    extracted_text = readers[extension](data)

    if not extracted_text.strip():
        raise BotError(
            "No readable text was found in this file. "
            "Scanned PDFs (images only) are not supported."
        )

    return extracted_text


# --------------------------------------------------------------------------
# 2. Chunking
# --------------------------------------------------------------------------

def _split_long_unit(unit: str, size: int) -> list[str]:
    """Break a long sentence at word boundaries."""

    pieces = []
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
    """Split text into sentences without crossing paragraphs."""

    units = []

    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = re.sub(r"\s+", " ", paragraph).strip()

        if not paragraph:
            continue

        sentences = re.split(r"(?<=[.!?])\s+", paragraph)

        for sentence in sentences:
            if len(sentence) > size:
                units.extend(
                    _split_long_unit(sentence, size)
                )
            else:
                units.append(sentence)

    return units


def _overlap_tail(chunk: str, overlap: int) -> str:
    """Return the last part of a chunk for context overlap."""

    if overlap <= 0 or len(chunk) <= overlap:
        return ""

    tail = chunk[-overlap:]
    first_space = tail.find(" ")

    return (
        tail[first_space + 1:]
        if first_space != -1
        else ""
    )


def chunk_text(
    text: str,
    size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[str]:
    """Split text into chunks with a small overlap."""

    chunks = []
    current = ""

    for unit in _split_into_units(text, size):
        if current and len(current) + 1 + len(unit) > size:
            chunks.append(current)

            tail = _overlap_tail(current, overlap)

            if tail and len(tail) + 1 + len(unit) <= size:
                current = f"{tail} {unit}"
            else:
                current = unit

        else:
            current = f"{current} {unit}".strip()

    if current:
        chunks.append(current)

    return chunks


# --------------------------------------------------------------------------
# 3. Embedding
# --------------------------------------------------------------------------

def embed_texts(
    texts: list[str],
    task_type: str,
) -> list[list[float]]:
    """Generate embeddings in batches."""

    if not texts:
        return []

    client = get_client()

    embed_config = types.EmbedContentConfig(
        task_type=task_type
    )

    vectors = []

    for start in range(0, len(texts), config.EMBED_BATCH_SIZE):
        batch = texts[
            start:start + config.EMBED_BATCH_SIZE
        ]

        response = call_with_retry(
            lambda batch=batch: client.models.embed_content(
                model=config.EMBED_MODEL,
                contents=batch,
                config=embed_config,
            )
        )

        embeddings = response.embeddings or []

        if len(embeddings) != len(batch):
            raise BotError(
                "The embedding service returned an unexpected response.",
                status_code=502,
            )

        vectors.extend(
            list(item.values)
            for item in embeddings
        )

    return vectors


# --------------------------------------------------------------------------
# 4. Storage - Qdrant Cloud
# --------------------------------------------------------------------------

def _point_id(filename: str, index: int) -> str:
    """Generate a stable UUID for each document chunk."""

    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"helppilot:{filename}:{index}",
        )
    )


def ingest_document(filename: str, data: bytes) -> int:
    """Extract, embed, and store a document in Qdrant."""

    name = Path(filename).name

    chunks = chunk_text(
        extract_text(name, data)
    )

    if not chunks:
        raise BotError(
            "No readable text was found in this file."
        )

    # Generate embeddings before replacing an existing document.
    vectors = embed_texts(
        chunks,
        "RETRIEVAL_DOCUMENT",
    )

    if not vectors:
        raise BotError(
            "The embedding service returned no vectors.",
            status_code=502,
        )

    ensure_collection(len(vectors[0]))

    client = get_qdrant_client()

    # Replace the previous version of this document.
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
            points=points[
                start:start + UPSERT_BATCH_SIZE
            ],
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

    query_vectors = embed_texts(
        [question],
        "RETRIEVAL_QUERY",
    )

    if not query_vectors:
        return []

    query_vector = query_vectors[0]

    result = client.query_points(
        collection_name=config.COLLECTION_NAME,
        query=query_vector,
        limit=min(config.TOP_K, total),
        with_payload=True,
    )

    chunks = []

    for point in result.points:
        payload = point.payload or {}

        similarity = float(point.score or 0.0)

        source = str(
            payload.get("source", "Unknown")
        )

        text = str(
            payload.get("text", "")
        )

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
        {
            "filename": name,
            "chunks": count,
        }
        for name, count in sorted(counts.items())
    ]


def delete_document(filename: str) -> int:
    """Delete all chunks belonging to a document."""

    client = get_qdrant_client()

    if not client.collection_exists(config.COLLECTION_NAME):
        return 0

    document_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="source",
                match=models.MatchValue(
                    value=filename
                ),
            )
        ]
    )

    result = client.count(
        collection_name=config.COLLECTION_NAME,
        count_filter=document_filter,
        exact=True,
    )

    if result.count:
        client.delete(
            collection_name=config.COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=document_filter
            ),
            wait=True,
        )

    return result.count

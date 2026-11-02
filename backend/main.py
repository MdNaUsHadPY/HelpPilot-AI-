"""FastAPI application: chat + document-management API and static frontend."""

import hmac
import logging
from typing import Literal

import uvicorn
from fastapi import Depends, FastAPI, File, Header, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

import config
import llm
import rag

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("support_bot.api")

app = FastAPI(title="Customer Support Bot")

GENERIC_ERROR = "Something went wrong on our side. Please try again in a moment."


# --------------------------------------------------------------------------
# Request models
# --------------------------------------------------------------------------

class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=5000)


class ChatRequest(BaseModel):
    message: str
    history: list[HistoryMessage] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Error handling: always return {"error": "..."} and never a stack trace
# --------------------------------------------------------------------------

def _error(message: str, status_code: int) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status_code)


@app.exception_handler(rag.BotError)
async def handle_bot_error(_: Request, exc: rag.BotError) -> JSONResponse:
    return _error(exc.message, exc.status_code)


@app.exception_handler(StarletteHTTPException)
async def handle_http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
    return _error(str(exc.detail), exc.status_code)


@app.exception_handler(RequestValidationError)
async def handle_validation_error(_: Request, __: RequestValidationError) -> JSONResponse:
    return _error("The request was not valid. Please check what you sent.", 422)


@app.exception_handler(Exception)
async def handle_unexpected_error(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error", exc_info=exc)
    return _error(GENERIC_ERROR, 500)


# --------------------------------------------------------------------------
# Optional admin protection
# --------------------------------------------------------------------------

def require_admin(x_admin_password: str | None = Header(default=None)) -> None:
    """If ADMIN_PASSWORD is set, the X-Admin-Password header must match it."""
    if not config.ADMIN_PASSWORD:
        return
    supplied = (x_admin_password or "").encode()
    if not hmac.compare_digest(supplied, config.ADMIN_PASSWORD.encode()):
        raise StarletteHTTPException(401, "Incorrect or missing admin password.")


# --------------------------------------------------------------------------
# API routes (plain `def` so blocking Gemini/Chroma calls run in a thread pool)
# --------------------------------------------------------------------------

@app.get("/api/health")
def health() -> dict[str, bool | str]:
    return {
        "status": "ok",
        "gemini_key_set": bool(config.GEMINI_API_KEY),
        "admin_protected": bool(config.ADMIN_PASSWORD),
    }


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict[str, str | list[str]]:
    question = request.message.strip()
    if not question:
        raise StarletteHTTPException(400, "Please type a question.")
    if len(question) > config.MAX_MESSAGE_CHARS:
        raise StarletteHTTPException(
            400, f"Your message is too long (maximum {config.MAX_MESSAGE_CHARS} characters)."
        )

    history = [message.model_dump() for message in request.history]
    answer, sources = llm.answer_question(question, history)
    return {"answer": answer, "sources": sources}


def _ingest_one(upload: UploadFile) -> dict[str, str | int]:
    """Index one uploaded file and describe the outcome."""
    filename = upload.filename or "unnamed"
    try:
        data = upload.file.read(config.MAX_FILE_BYTES + 1)
        if len(data) > config.MAX_FILE_BYTES:
            raise rag.BotError(
                f"The file is too large (maximum {config.MAX_FILE_BYTES // (1024 * 1024)} MB)."
            )
        return {"filename": filename, "chunks": rag.ingest_document(filename, data)}
    except rag.BotError as exc:
        return {"filename": filename, "error": exc.message}
    except Exception:  # keep one bad file from failing the whole batch
        logger.exception("Failed to ingest %s", filename)
        return {"filename": filename, "error": GENERIC_ERROR}


@app.post("/api/ingest", dependencies=[Depends(require_admin)])
def ingest(files: list[UploadFile] = File(...)) -> dict[str, list[dict[str, str | int]]]:
    return {"results": [_ingest_one(upload) for upload in files]}


@app.get("/api/documents", dependencies=[Depends(require_admin)])
def documents() -> dict[str, list[dict[str, str | int]]]:
    return {"documents": rag.list_documents()}


@app.delete("/api/documents/{filename}", dependencies=[Depends(require_admin)])
def delete_document(filename: str) -> dict[str, str | int]:
    removed = rag.delete_document(filename)
    if removed == 0:
        raise StarletteHTTPException(404, "That document was not found.")
    return {"filename": filename, "chunks_removed": removed}


# --------------------------------------------------------------------------
# Frontend (registered last so it never shadows the API routes)
# --------------------------------------------------------------------------

@app.get("/admin", include_in_schema=False)
def admin_page() -> FileResponse:
    return FileResponse(config.FRONTEND_DIR / "admin.html")


app.mount("/", StaticFiles(directory=config.FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    # 0.0.0.0 makes the server reachable from Replit's preview and the internet.
    uvicorn.run(app, host="0.0.0.0", port=config.PORT)

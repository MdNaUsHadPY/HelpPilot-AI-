
"""Answer generation: build the prompt from retrieved chunks and ask Gemini."""

import logging
import time
from typing import Mapping

from google.genai import types

import config
import rag

logger = logging.getLogger("support_bot.llm")


SYSTEM_INSTRUCTION = (
    "You are a polite, friendly customer support assistant for a small business. "
    "Answer ONLY using the provided context. If the answer is not in the context, "
    "say you don't have that information and suggest contacting the business "
    "directly. Never invent prices, policies, dates or contact details. "
    "Keep answers short and clear. Reply in plain text without markdown formatting."
)

NO_INFO_MESSAGE = (
    "I'm sorry, I don't have information on that. "
    "Please contact the business directly and they'll be happy to help."
)

# The chat page uses "assistant"; Gemini calls the same role "model".
GEMINI_ROLES = {"user": "user", "assistant": "model"}


def _recent_history(
    history: list[Mapping[str, str]]
) -> list[types.Content]:
    """Convert recent chat messages into Gemini content objects."""

    recent = history[-config.MAX_HISTORY_TURNS:]

    # Gemini conversations must begin with a user message.
    while recent and recent[0]["role"] != "user":
        recent = recent[1:]

    return [
        types.Content(
            role=GEMINI_ROLES[message["role"]],
            parts=[types.Part(text=message["content"])],
        )
        for message in recent
        if message["role"] in GEMINI_ROLES
    ]


def _format_context(chunks: list[rag.RetrievedChunk]) -> str:
    """Format retrieved document chunks."""

    return "\n\n".join(
        f"[{number}] (from {chunk.source})\n{chunk.text}"
        for number, chunk in enumerate(chunks, start=1)
    )


def build_contents(
    question: str,
    chunks: list[rag.RetrievedChunk],
    history: list[Mapping[str, str]],
) -> list[types.Content]:
    """Build Gemini conversation contents."""

    final_message = (
        f"Context:\n{_format_context(chunks)}\n\n"
        f"Customer question: {question}"
    )

    return _recent_history(history) + [
        types.Content(
            role="user",
            parts=[types.Part(text=final_message)],
        )
    ]


def _unique_sources(
    chunks: list[rag.RetrievedChunk]
) -> list[str]:
    """Return unique source filenames."""

    return list(dict.fromkeys(chunk.source for chunk in chunks))


def answer_question(
    question: str,
    history: list[Mapping[str, str]],
) -> tuple[str, list[str]]:
    """Retrieve documents and generate a customer support answer."""

    start = time.perf_counter()

    # Step 1: Retrieve relevant documents.
    retrieval_start = time.perf_counter()

    chunks = rag.retrieve(question)

    retrieval_time = time.perf_counter() - retrieval_start

    logger.info(
        "Document retrieval took %.2f seconds",
        retrieval_time,
    )

    if not chunks:
        logger.info(
            "No relevant documents found. Total time: %.2f seconds",
            time.perf_counter() - start,
        )
        return NO_INFO_MESSAGE, []

    # Step 2: Generate answer using Gemini.
    client = rag.get_client()

    generation_start = time.perf_counter()

    response = rag.call_with_retry(
        lambda: client.models.generate_content(
            model=config.GEN_MODEL,
            contents=build_contents(question, chunks, history),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=config.GENERATION_TEMPERATURE,
                max_output_tokens=config.MAX_OUTPUT_TOKENS,
                thinking_config=types.ThinkingConfig(
                    thinking_budget=config.GEN_THINKING_BUDGET
                ),
            ),
        )
    )

    generation_time = time.perf_counter() - generation_start

    logger.info(
        "Gemini generation took %.2f seconds",
        generation_time,
    )

    answer = (response.text or "").strip()

    if not answer:
        raise rag.BotError(
            "The assistant could not produce an answer. Please rephrase your question.",
            status_code=502,
        )

    total_time = time.perf_counter() - start

    logger.info(
        "Total answer time: %.2f seconds",
        total_time,
    )

    return answer, _unique_sources(chunks)

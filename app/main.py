"""
main.py — FastAPI application entry point for the SHL assessment agent.

Endpoints
─────────
GET  /health   → {"status": "ok"}
POST /chat     → ChatResponse (stateless; full history sent per request)
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from app.models import ChatRequest, ChatResponse
from app.catalog_loader import load_catalog
from app.retriever import HybridRetriever
from app.agent import generate_reply

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

catalog: list = []
retriever: HybridRetriever | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load catalog and build retriever index at startup."""
    global catalog, retriever
    logger.info("Loading SHL catalog …")
    catalog = load_catalog()
    logger.info(f"Catalog loaded: {len(catalog)} items")
    logger.info("Building HybridRetriever index …")
    retriever = HybridRetriever(catalog)
    logger.info("Retriever ready.")
    yield


app = FastAPI(
    title="SHL Assessment Agent",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        messages = [m.dict() for m in request.messages]

        user_turns = [m["content"] for m in messages if m["role"] == "user"]
        latest_user_msg = user_turns[-1] if user_turns else ""
        conversation_context = " ".join(user_turns)

        retrieved = retriever.search(
            query=latest_user_msg,
            conversation_context=conversation_context,
            top_k=10,
        )

        response = generate_reply(messages, retrieved)
        return response

    except Exception as exc:
        logger.exception(f"Unhandled error in /chat: {exc}")
        return ChatResponse(
            reply=(
                "I'm having trouble processing your request right now. "
                "Please try again in a moment."
            ),
            recommendations=[],
            end_of_conversation=False,
        )



@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "reply": "An unexpected error occurred.",
            "recommendations": [],
            "end_of_conversation": False,
        },
    )
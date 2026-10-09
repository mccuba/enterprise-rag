"""Enterprise RAG Assistant — FastAPI entrypoint."""
from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import get_settings
from app.db.session import create_engine_and_session, init_db
from app.providers.embeddings import build_embedding_provider
from app.providers.llm import build_llm_provider
from app.providers.ollama_client import RotatingOllamaClient
from app.providers.vector_store import build_vector_store


@dataclass
class AppState:
    llm: Any
    embeddings: Any
    vector_store: Any
    engine: Any
    session_factory: Any


app_state = AppState(
    llm=None,
    embeddings=None,
    vector_store=None,
    engine=None,
    session_factory=None,
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings = get_settings()
    engine, session_factory = create_engine_and_session(settings)
    await init_db(engine)

    ollama = RotatingOllamaClient(settings.ollama_host, settings.token_list)
    embeddings = build_embedding_provider(settings, ollama)
    vector_store = build_vector_store(settings, embeddings)
    llm = build_llm_provider(settings, ollama)
    # Chat = Ollama Cloud; embeddings = FastEmbed local por defecto.

    app_state.llm = llm
    app_state.embeddings = embeddings
    app_state.vector_store = vector_store
    app_state.engine = engine
    app_state.session_factory = session_factory
    yield
    await engine.dispose()


app = FastAPI(
    title="Enterprise RAG Assistant",
    description="Prototipo RAG con grounding y trazabilidad de fuentes",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

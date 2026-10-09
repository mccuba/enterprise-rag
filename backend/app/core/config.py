"""Configuración centralizada vía variables de entorno."""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    ollama_host: str = "https://ollama.com"
    ollama_model: str = "gpt-oss:20b"
    ollama_tokens: str = ""

    # fastembed (local ONNX) | ollama (solo si el cloud expone /api/embed)
    embedding_provider: str = "fastembed"
    # Modelo FastEmbed por defecto (multilingüe, liviano). Alt: BAAI/bge-small-en-v1.5
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    database_url: str = "sqlite+aiosqlite:///./data/rag.db"
    chroma_path: str = "./data/chroma"
    upload_dir: str = "./data/uploads"

    chunk_size: int = 800
    chunk_overlap: int = 160
    # Vecinos en retrieval (valores ~5–7 son habituales en labs RAG).
    top_k: int = 7
    min_relevance: float = 0.25
    # Baja → menos alucinación en respuestas grounded (RAG factual).
    llm_temperature: float = 0.2
    # Libros de 500+ pág. saturan CPU en pymupdf4llm + FastEmbed (demo local).
    max_pdf_pages: int = 80

    @property
    def token_list(self) -> list[str]:
        tokens = [t.strip() for t in self.ollama_tokens.split(",") if t.strip()]
        if not tokens:
            raise ValueError(
                "OLLAMA_TOKENS vacío. Define al menos un token en .env "
                "(ver .env.example)."
            )
        return tokens


@lru_cache
def get_settings() -> Settings:
    return Settings()

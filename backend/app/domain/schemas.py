"""Contratos Pydantic de la API."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class SourceCitation(BaseModel):
    citation_index: int = Field(..., description="Número visible en la UI, ej. [1]")
    chunk_id: str
    document_id: str
    filename: str
    page: int | None = None
    text: str
    score: float


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    session_id: str = Field(..., min_length=1, max_length=64)


class AskResponse(BaseModel):
    session_id: str
    answer: str
    # answer con marcadores [1], [2] listos para el FE
    answer_with_citations: str
    grounded: bool
    insufficient_context: bool
    sources: list[SourceCitation]
    used_chunk_ids: list[str]


class HistoryMessage(BaseModel):
    role: str
    content: str
    sources: list[dict[str, Any]] | None = None
    created_at: datetime | None = None


class HistoryResponse(BaseModel):
    session_id: str
    messages: list[HistoryMessage]


class DocumentInfo(BaseModel):
    id: str
    filename: str
    pages: int
    chunk_count: int
    created_at: datetime | None = None


class DocumentUploadResponse(BaseModel):
    document: DocumentInfo
    message: str


class DocumentTextResponse(BaseModel):
    id: str
    filename: str
    full_text: str
    chunks: list[dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    llm_model: str
    documents: int

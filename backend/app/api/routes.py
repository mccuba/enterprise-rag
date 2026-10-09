"""Endpoints FastAPI del reto."""
from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import func, select

from app.core.config import Settings, get_settings
from app.db.models import ChunkRow, DocumentRow, MessageRow
from app.domain.schemas import (
    AskRequest,
    AskResponse,
    DocumentInfo,
    DocumentTextResponse,
    DocumentUploadResponse,
    HealthResponse,
    HistoryMessage,
    HistoryResponse,
)
from app.services.ingest import ingest_pdf
from app.services.rag import RagService

router = APIRouter()


def get_deps():
    """Inyectado desde main via app.state."""
    from app.main import app_state

    return app_state


@router.get("/health", response_model=HealthResponse)
async def health(
    settings: Annotated[Settings, Depends(get_settings)],
):
    state = get_deps()
    async with state.session_factory() as session:
        count = await session.scalar(select(func.count()).select_from(DocumentRow))
    return HealthResponse(
        status="ok",
        llm_model=settings.ollama_model,
        documents=int(count or 0),
    )


@router.post("/documents", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Solo se aceptan archivos PDF.")

    state = get_deps()
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    dest = Path(settings.upload_dir) / f"{uuid.uuid4().hex}_{file.filename}"
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    try:
        async with state.session_factory() as session:
            doc = await ingest_pdf(
                file_path=dest,
                filename=file.filename,
                session=session,
                vector_store=state.vector_store,
                settings=settings,
            )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"Error al procesar PDF: {exc}") from exc

    return DocumentUploadResponse(
        document=DocumentInfo(
            id=doc.id,
            filename=doc.filename,
            pages=doc.pages,
            chunk_count=doc.chunk_count,
            created_at=doc.created_at,
        ),
        message=f"Documento indexado: {doc.chunk_count} chunks.",
    )


@router.get("/documents", response_model=list[DocumentInfo])
async def list_documents():
    state = get_deps()
    async with state.session_factory() as session:
        rows = (await session.scalars(select(DocumentRow).order_by(DocumentRow.created_at.desc()))).all()
    return [
        DocumentInfo(
            id=r.id,
            filename=r.filename,
            pages=r.pages,
            chunk_count=r.chunk_count,
            created_at=r.created_at,
        )
        for r in rows
    ]


@router.get("/documents/{document_id}/text", response_model=DocumentTextResponse)
async def get_document_text(document_id: str):
    """Texto completo + chunks para resaltar citas en el frontend."""
    state = get_deps()
    async with state.session_factory() as session:
        doc = await session.get(DocumentRow, document_id)
        if not doc:
            raise HTTPException(404, "Documento no encontrado")
        chunks = (
            await session.scalars(
                select(ChunkRow)
                .where(ChunkRow.document_id == document_id)
                .order_by(ChunkRow.chunk_index)
            )
        ).all()
    return DocumentTextResponse(
        id=doc.id,
        filename=doc.filename,
        full_text=doc.full_text,
        chunks=[
            {
                "chunk_id": c.id,
                "chunk_index": c.chunk_index,
                "page": c.page,
                "text": c.text,
                "char_start": c.char_start,
                "char_end": c.char_end,
            }
            for c in chunks
        ],
    )


@router.post("/ask", response_model=AskResponse)
async def ask(payload: AskRequest, settings: Settings = Depends(get_settings)):
    if not payload.question.strip():
        raise HTTPException(400, "La pregunta no puede estar vacía.")
    state = get_deps()
    rag = RagService(llm=state.llm, vector_store=state.vector_store, settings=settings)
    async with state.session_factory() as session:
        return await rag.ask(
            question=payload.question.strip(),
            session_id=payload.session_id,
            db=session,
        )


@router.get("/history/{session_id}", response_model=HistoryResponse)
async def get_history(session_id: str):
    state = get_deps()
    async with state.session_factory() as session:
        rows = (
            await session.scalars(
                select(MessageRow)
                .where(MessageRow.session_id == session_id)
                .order_by(MessageRow.id.asc())
            )
        ).all()
    messages = []
    for r in rows:
        sources = json.loads(r.sources_json) if r.sources_json else None
        messages.append(
            HistoryMessage(
                role=r.role,
                content=r.content,
                sources=sources,
                created_at=r.created_at,
            )
        )
    return HistoryResponse(session_id=session_id, messages=messages)

"""Ingesta de PDFs: pymupdf4llm → Markdown → chunking → vector store + SQL.

PDF→Markdown suele preservar mejor títulos/tablas que extractores PDF crudos.
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path

import fitz
import pymupdf4llm
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.db.models import ChunkRow, DocumentRow
from app.providers.vector_store import VectorStore
from app.services.chunking import split_pages, split_text

logger = logging.getLogger(__name__)


def count_pdf_pages(path: Path) -> int:
    with fitz.open(str(path)) as doc:
        return doc.page_count


def pdf_to_markdown(path: Path) -> tuple[list[tuple[int, str]], str]:
    """Convierte PDF a Markdown.

    Intenta `page_chunks=True` para conservar número de página (trazabilidad).
    Fallback: un solo markdown completo.
    """
    try:
        page_chunks = pymupdf4llm.to_markdown(str(path), page_chunks=True)
    except TypeError:
        page_chunks = None

    if isinstance(page_chunks, list) and page_chunks:
        pages: list[tuple[int, str]] = []
        parts: list[str] = []
        for i, item in enumerate(page_chunks, start=1):
            if isinstance(item, dict):
                md = str(item.get("text") or item.get("markdown") or "")
                page_num = int(item.get("metadata", {}).get("page", i))
                # pymupdf a veces usa 0-based
                if page_num == 0:
                    page_num = i
            else:
                md = str(item)
                page_num = i
            pages.append((page_num, md))
            parts.append(f"\n\n--- Página {page_num} ---\n\n{md}")
        full = "".join(parts).strip()
        return pages, full

    md_text = pymupdf4llm.to_markdown(str(path))
    return [(1, md_text)], md_text.strip()


async def ingest_pdf(
    *,
    file_path: Path,
    filename: str,
    session: AsyncSession,
    vector_store: VectorStore,
    settings: Settings,
) -> DocumentRow:
    document_id = uuid.uuid4().hex
    n_pages = count_pdf_pages(file_path)
    if n_pages > settings.max_pdf_pages:
        raise ValueError(
            f"El PDF tiene {n_pages} páginas; el máximo permitido es "
            f"{settings.max_pdf_pages}. Usa un PDF más corto (≥10 pág. del reto "
            f"basta; ej. un paper arXiv). Un libro de 500+ pág. tarda mucho en "
            f"Markdown+embeddings en CPU."
        )
    logger.info("Ingesta %s: %d páginas → Markdown…", filename, n_pages)
    pages, full_text = pdf_to_markdown(file_path)
    logger.info("Ingesta %s: Markdown listo, chunking + embeddings…", filename)

    # Guardar .md junto al PDF para inspección / resaltado en el frontend
    md_path = file_path.with_suffix(".md")
    md_path.write_text(full_text, encoding="utf-8")

    if len(pages) == 1 and pages[0][0] == 1 and "--- Página" not in full_text[:80]:
        text_chunks = split_text(
            full_text,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            page=None,
        )
        page_count = max(1, full_text.count("--- Página"))
    else:
        text_chunks = split_pages(
            pages,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        page_count = len(pages)

    if not text_chunks:
        raise ValueError("No se pudo extraer texto del PDF (¿escaneado sin OCR?).")

    doc = DocumentRow(
        id=document_id,
        filename=filename,
        title=filename,
        pages=page_count,
        chunk_count=len(text_chunks),
        full_text=full_text,
    )
    session.add(doc)

    payload = []
    for ch in text_chunks:
        chunk_id = f"{document_id}_{ch.chunk_index:04d}"
        session.add(
            ChunkRow(
                id=chunk_id,
                document_id=document_id,
                chunk_index=ch.chunk_index,
                page=ch.page,
                text=ch.text,
                char_start=ch.char_start,
                char_end=ch.char_end,
            )
        )
        payload.append(
            {
                "chunk_id": chunk_id,
                "chunk_index": ch.chunk_index,
                "page": ch.page,
                "text": ch.text,
            }
        )

    vector_store.upsert_chunks(
        document_id=document_id,
        filename=filename,
        chunks=payload,
    )
    await session.commit()
    await session.refresh(doc)
    return doc

"""Estrategia de chunking sobre texto/Markdown."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TextChunk:
    chunk_index: int
    text: str
    page: int | None
    char_start: int
    char_end: int


def split_text(
    text: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
    page: int | None = None,
    start_index: int = 0,
) -> list[TextChunk]:
    """Divide texto/Markdown con overlap ~20%."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap debe ser menor que chunk_size")

    text = text.strip()
    if not text:
        return []

    chunks: list[TextChunk] = []
    idx = start_index
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            window = text[start:end]
            # Priorizar cortes de Markdown / párrafo
            for sep in ("\n\n", "\n#", "\n", ". ", " "):
                pos = window.rfind(sep)
                if pos > chunk_size * 0.4:
                    end = start + pos + (len(sep) if sep != "\n#" else 1)
                    break
        piece = text[start:end].strip()
        if piece:
            chunks.append(
                TextChunk(
                    chunk_index=idx,
                    text=piece,
                    page=page,
                    char_start=start,
                    char_end=end,
                )
            )
            idx += 1
        if end >= len(text):
            break
        start = max(0, end - chunk_overlap)
    return chunks


def split_pages(
    pages: list[tuple[int, str]],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[TextChunk]:
    """Compat: chunking por página (page, text)."""
    chunks: list[TextChunk] = []
    for page_num, page_text in pages:
        part = split_text(
            page_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            page=page_num,
            start_index=len(chunks),
        )
        chunks.extend(part)
    return chunks

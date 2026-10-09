"""Puerto e implementación Chroma para búsqueda vectorial."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import app.core.sqlite_patch  # noqa: F401  — antes de chromadb (sqlite >= 3.35)
import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import Settings
from app.providers.embeddings import EmbeddingProvider


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str
    filename: str
    page: int | None
    text: str
    score: float
    citation_index: int = 0


class VectorStore(ABC):
    @abstractmethod
    def upsert_chunks(
        self,
        *,
        document_id: str,
        filename: str,
        chunks: list[dict[str, Any]],
    ) -> int:
        raise NotImplementedError

    @abstractmethod
    def similarity_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        raise NotImplementedError

    @abstractmethod
    def delete_document(self, document_id: str) -> None:
        raise NotImplementedError


class ChromaVectorStore(VectorStore):
    COLLECTION = "enterprise_rag"

    def __init__(self, path: str, embeddings: EmbeddingProvider):
        Path(path).mkdir(parents=True, exist_ok=True)
        self._embeddings = embeddings
        self._client = chromadb.PersistentClient(
            path=path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(
        self,
        *,
        document_id: str,
        filename: str,
        chunks: list[dict[str, Any]],
    ) -> int:
        if not chunks:
            return 0
        ids = [c["chunk_id"] for c in chunks]
        texts = [c["text"] for c in chunks]
        metadatas = [
            {
                "document_id": document_id,
                "filename": filename,
                "page": c.get("page") if c.get("page") is not None else -1,
                "chunk_index": c["chunk_index"],
            }
            for c in chunks
        ]
        vectors = self._embeddings.embed(texts)
        self._collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=vectors,
            metadatas=metadatas,
        )
        return len(ids)

    def similarity_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        if self._collection.count() == 0:
            return []
        qvec = self._embeddings.embed([query])[0]
        result = self._collection.query(
            query_embeddings=[qvec],
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        chunks: list[RetrievedChunk] = []
        ids = result["ids"][0]
        docs = result["documents"][0]
        metas = result["metadatas"][0]
        dists = result["distances"][0]
        for i, chunk_id in enumerate(ids):
            # Chroma cosine distance: 0 = idéntico. score = 1 - distance.
            distance = float(dists[i])
            score = max(0.0, 1.0 - distance)
            page_raw = metas[i].get("page", -1)
            page = None if page_raw in (-1, None) else int(page_raw)
            chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    document_id=str(metas[i].get("document_id", "")),
                    filename=str(metas[i].get("filename", "")),
                    page=page,
                    text=docs[i],
                    score=score,
                    citation_index=i + 1,
                )
            )
        return chunks

    def delete_document(self, document_id: str) -> None:
        self._collection.delete(where={"document_id": document_id})


def build_vector_store(settings: Settings, embeddings: EmbeddingProvider) -> VectorStore:
    return ChromaVectorStore(settings.chroma_path, embeddings)

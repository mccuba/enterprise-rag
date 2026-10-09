"""Orquestación RAG: retrieve → ground → generate → cite."""
from __future__ import annotations

import json
import re
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.db.models import MessageRow
from app.domain.schemas import AskResponse, SourceCitation
from app.providers.llm import LLMProvider
from app.providers.vector_store import RetrievedChunk, VectorStore

SYSTEM_PROMPT = """Eres un asistente RAG. Respondes SOLO con la información del CONTEXTO.

Reglas estrictas:
1. Fundamenta cada afirmación en el contexto. Usa citas [n] justo después del dato citado.
2. Solo puedes usar números de cita que existan en el contexto (ej. [1], [2]).
3. Si el contexto NO contiene información suficiente para responder, responde exactamente:
   INSUFFICIENT_CONTEXT: <breve explicación de qué falta>
   No inventes hechos, cifras ni conclusiones.
4. No uses conocimiento externo al contexto.
5. Responde en el mismo idioma de la pregunta del usuario.
6. Sé claro y conciso.
"""


def _build_context_block(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for c in chunks:
        page = f", página {c.page}" if c.page else ""
        parts.append(
            f"[{c.citation_index}] (id={c.chunk_id}, archivo={c.filename}{page}, "
            f"score={c.score:.3f})\n{c.text}"
        )
    return "\n\n".join(parts)


def _filter_by_relevance(
    chunks: list[RetrievedChunk], min_relevance: float
) -> list[RetrievedChunk]:
    kept = [c for c in chunks if c.score >= min_relevance]
    # Reindexar citas 1..n sobre los que pasan el umbral
    for i, c in enumerate(kept, start=1):
        c.citation_index = i
    return kept


def _extract_used_citations(answer: str, max_index: int) -> list[int]:
    found = sorted({int(m) for m in re.findall(r"\[(\d+)\]", answer)})
    return [n for n in found if 1 <= n <= max_index]


class RagService:
    def __init__(
        self,
        *,
        llm: LLMProvider,
        vector_store: VectorStore,
        settings: Settings,
    ):
        self.llm = llm
        self.vector_store = vector_store
        self.settings = settings

    async def ask(
        self,
        *,
        question: str,
        session_id: str,
        db: AsyncSession,
    ) -> AskResponse:
        raw = self.vector_store.similarity_search(question, self.settings.top_k)
        chunks = _filter_by_relevance(raw, self.settings.min_relevance)

        if not chunks:
            answer = (
                "No encontré información suficiente en los documentos cargados "
                "para responder esa pregunta."
            )
            response = AskResponse(
                session_id=session_id,
                answer=answer,
                answer_with_citations=answer,
                grounded=False,
                insufficient_context=True,
                sources=[],
                used_chunk_ids=[],
            )
            await self._persist(db, session_id, question, response)
            return response

        context = _build_context_block(chunks)
        user_prompt = (
            f"CONTEXTO:\n{context}\n\n"
            f"PREGUNTA:\n{question}\n\n"
            "Respuesta (con citas [n] cuando corresponda):"
        )
        raw_answer = self.llm.chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )

        insufficient = raw_answer.startswith("INSUFFICIENT_CONTEXT")
        if insufficient:
            clean = raw_answer.replace("INSUFFICIENT_CONTEXT:", "", 1).strip()
            answer = (
                "No hay información suficiente en los documentos para responder "
                f"con certeza. {clean}"
            ).strip()
            sources: list[SourceCitation] = []
            used_ids: list[str] = []
            answer_with_citations = answer
            grounded = False
        else:
            used_nums = _extract_used_citations(raw_answer, len(chunks))
            # Si el modelo no citó, adjuntamos las fuentes recuperadas al final
            answer_with_citations = raw_answer
            if not used_nums:
                used_nums = [c.citation_index for c in chunks]
                footer = " ".join(f"[{n}]" for n in used_nums)
                answer_with_citations = f"{raw_answer.strip()} {footer}".strip()

            by_idx = {c.citation_index: c for c in chunks}
            sources = [
                SourceCitation(
                    citation_index=n,
                    chunk_id=by_idx[n].chunk_id,
                    document_id=by_idx[n].document_id,
                    filename=by_idx[n].filename,
                    page=by_idx[n].page,
                    text=by_idx[n].text,
                    score=by_idx[n].score,
                )
                for n in used_nums
                if n in by_idx
            ]
            used_ids = [s.chunk_id for s in sources]
            # Limpiar marcadores para el campo "answer" plano
            answer = re.sub(r"\s*\[\d+\]", "", answer_with_citations).strip()
            grounded = True

        response = AskResponse(
            session_id=session_id,
            answer=answer,
            answer_with_citations=answer_with_citations,
            grounded=grounded,
            insufficient_context=insufficient or not grounded,
            sources=sources,
            used_chunk_ids=used_ids,
        )
        await self._persist(db, session_id, question, response)
        return response

    async def _persist(
        self,
        db: AsyncSession,
        session_id: str,
        question: str,
        response: AskResponse,
    ) -> None:
        sources_payload: list[dict[str, Any]] = [
            s.model_dump() for s in response.sources
        ]
        db.add(MessageRow(session_id=session_id, role="user", content=question))
        db.add(
            MessageRow(
                session_id=session_id,
                role="assistant",
                content=response.answer_with_citations,
                sources_json=json.dumps(sources_payload, ensure_ascii=False),
            )
        )
        await db.commit()

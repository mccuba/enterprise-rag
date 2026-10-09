"""Cliente Ollama Cloud compartido (chat + embeddings) con rotación de keys."""
from __future__ import annotations

from typing import Any

from ollama import Client, ResponseError

_ROTATE_ON_STATUS = {401, 403, 429, 500}


class RotatingOllamaClient:
    def __init__(self, host: str, tokens: list[str]):
        if not tokens:
            raise ValueError("Se necesita al menos un OLLAMA token")
        self.host = host
        self.tokens = list(tokens)
        self.idx = 0
        self._client = self._build()

    def _build(self) -> Client:
        return Client(
            host=self.host,
            headers={"Authorization": f"Bearer {self.tokens[self.idx]}"},
        )

    def _rotate(self, status: int | None) -> None:
        self.idx = (self.idx + 1) % len(self.tokens)
        self._client = self._build()

    def _call(self, fn_name: str, **kwargs: Any) -> Any:
        last_err: Exception | None = None
        for attempt in range(len(self.tokens)):
            try:
                fn = getattr(self._client, fn_name)
                return fn(**kwargs)
            except ResponseError as exc:
                last_err = exc
                status = getattr(exc, "status_code", None)
                if status not in _ROTATE_ON_STATUS or attempt + 1 >= len(self.tokens):
                    raise
                self._rotate(status)
        assert last_err is not None
        raise last_err

    def chat(self, *, model: str, messages: list[dict[str, str]], **kwargs: Any) -> str:
        response = self._call("chat", model=model, messages=messages, **kwargs)
        content = response.get("message", {}).get("content", "")
        return str(content).strip()

    def embed(self, *, model: str, texts: list[str]) -> list[list[float]]:
        """Usa solo /api/embed (Ollama Cloud no tiene /api/embeddings)."""
        response = self._call("embed", model=model, input=texts)
        vectors = response.get("embeddings")
        if not vectors:
            raise RuntimeError(
                f"Ollama embed no devolvió vectors para model={model}. "
                "Tu plan/cloud puede no incluir modelos de embedding."
            )
        return [list(map(float, v)) for v in vectors]

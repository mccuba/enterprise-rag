"""Proveedor de embeddings desacoplado.

Por defecto: FastEmbed (ONNX/CPU, sin PyTorch/CUDA).
Ollama Cloud puede no exponer modelos de embedding (/api/embed 401,
/api/embeddings 404); en ese caso se usa FastEmbed para indexar.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from app.core.config import Settings
from app.providers.ollama_client import RotatingOllamaClient


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    @property
    @abstractmethod
    def dimension(self) -> int:
        raise NotImplementedError


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = sum(x * x for x in vec) ** 0.5
    if norm <= 0:
        return vec
    return [x / norm for x in vec]


class FastEmbedProvider(EmbeddingProvider):
    """Embeddings locales vía ONNX — camino práctico sin GPU ni cupo Ollama."""

    def __init__(self, model_name: str):
        from fastembed import TextEmbedding

        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name)
        probe = next(self._model.embed(["dimension probe"]))
        self._dim = len(probe)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = [list(map(float, v)) for v in self._model.embed(texts)]
        return [_l2_normalize(v) for v in vectors]

    @property
    def dimension(self) -> int:
        return self._dim


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Solo si el proveedor Ollama expone /api/embed para el modelo elegido."""

    def __init__(self, client: RotatingOllamaClient, model: str):
        self._client = client
        self.model = model
        probe = self._client.embed(model=self.model, texts=["dimension probe"])
        self._dim = len(probe[0])

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [_l2_normalize(v) for v in self._client.embed(model=self.model, texts=texts)]

    @property
    def dimension(self) -> int:
        return self._dim


def build_embedding_provider(
    settings: Settings,
    client: RotatingOllamaClient | None = None,
) -> EmbeddingProvider:
    provider = settings.embedding_provider.lower().strip()
    if provider == "ollama":
        if client is None:
            raise ValueError("OllamaEmbeddingProvider requiere cliente Ollama")
        return OllamaEmbeddingProvider(client, settings.embedding_model)
    # default / fastembed
    return FastEmbedProvider(settings.embedding_model)

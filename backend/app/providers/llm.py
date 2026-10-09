"""Proveedor LLM desacoplado — implementación Ollama Cloud."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.core.config import Settings
from app.providers.ollama_client import RotatingOllamaClient


class LLMProvider(ABC):
    @abstractmethod
    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        raise NotImplementedError


class OllamaLLMProvider(LLMProvider):
    def __init__(self, client: RotatingOllamaClient, model: str, temperature: float = 0.2):
        self._client = client
        self.model = model
        self.temperature = temperature

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        options = dict(kwargs.pop("options", None) or {})
        options.setdefault("temperature", self.temperature)
        return self._client.chat(
            model=self.model,
            messages=messages,
            options=options,
            **kwargs,
        )


def build_llm_provider(settings: Settings, client: RotatingOllamaClient) -> LLMProvider:
    return OllamaLLMProvider(
        client,
        settings.ollama_model,
        temperature=settings.llm_temperature,
    )

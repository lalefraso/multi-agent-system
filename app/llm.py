from __future__ import annotations

import logging
import os
import time
import tomllib
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest
from langchain.chat_models import init_chat_model


ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
SETTINGS_FILE = Path(__file__).resolve().parents[1] / "settings.toml"
LOGGER = logging.getLogger(__name__)


class LLMConfigurationError(RuntimeError):
    """Raised when a configured LLM provider cannot be initialized."""


def load_llm_settings() -> dict[str, Any]:
    try:
        with SETTINGS_FILE.open("rb") as settings_file:
            config = tomllib.load(settings_file)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise LLMConfigurationError(
            f"Could not load LLM settings from {SETTINGS_FILE}: {exc}"
        ) from exc

    llm_settings = config.get("llm")
    if not isinstance(llm_settings, dict):
        raise LLMConfigurationError(
            f"Missing [llm] section in {SETTINGS_FILE}"
        )
    model = llm_settings.get("model")
    fallbacks = llm_settings.get("fallback_models")
    if not isinstance(model, str) or not model.strip():
        raise LLMConfigurationError(
            f"[llm].model must be a non-empty provider:model string in {SETTINGS_FILE}"
        )
    if not isinstance(fallbacks, list) or not all(
        isinstance(fallback, str) and fallback.strip() for fallback in fallbacks
    ):
        raise LLMConfigurationError(
            f"[llm].fallback_models must be a list of model identifiers in {SETTINGS_FILE}"
        )
    return llm_settings


def _create_chat_model(model_id: str, thinking_level: str | None = None) -> Any:
    if model_id == "baseten" or model_id.startswith("baseten:"):
        configured_model = model_id.partition(":")[2]
        model_name = os.getenv("BASETEN_MODEL") or configured_model
        required = ("BASETEN_API_KEY", "BASETEN_BASE_URL")
        missing = [name for name in required if not os.getenv(name)]
        if not model_name:
            missing.append("BASETEN_MODEL or baseten:<model>")
        if missing:
            raise LLMConfigurationError(
                "Missing required Baseten configuration: " + ", ".join(missing)
            )
        return init_chat_model(
            model=f"openai:{model_name}",
            base_url=os.environ["BASETEN_BASE_URL"],
            api_key=os.environ["BASETEN_API_KEY"],
            timeout=30,
            max_retries=1,
        )

    if ":" not in model_id:
        raise LLMConfigurationError(
            f"Invalid model identifier {model_id!r}; expected provider:model"
        )
    provider, _ = model_id.split(":", maxsplit=1)
    model_options: dict[str, Any] = {"timeout": 30, "max_retries": 1}
    if provider == "google_genai" and thinking_level:
        model_options["thinking_level"] = thinking_level
    return init_chat_model(model=model_id, **model_options)


class ProviderFallbackMiddleware(AgentMiddleware):
    def __init__(
        self,
        primary_model_name: str,
        fallback_models: list[tuple[str, Callable[[], Any]]],
        allowed_tool_names: set[str],
    ) -> None:
        super().__init__()
        self.primary_model_name = primary_model_name
        self.fallback_models = fallback_models
        self.allowed_tool_names = allowed_tool_names

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Any],
    ) -> Any:
        allowed_tools = [
            tool
            for tool in request.tools or []
            if self._tool_name(tool) in self.allowed_tool_names
        ]
        request = request.override(tools=allowed_tools)
        errors: list[Exception] = []
        model_factories = [
            (self.primary_model_name, lambda: request.model),
            *self.fallback_models,
        ]
        for attempt, (model_name, model_factory) in enumerate(model_factories, start=1):
            started_at = time.monotonic()
            LOGGER.info(
                "Calling model provider %s (attempt %d, messages=%d)",
                model_name,
                attempt,
                len(request.messages),
            )
            try:
                model = model_factory()
                response = handler(request.override(model=model))
            except Exception as exc:
                errors.append(exc)
                LOGGER.warning(
                    "Model provider %s failed (attempt %d, error_type=%s, duration_ms=%.0f)",
                    model_name,
                    attempt,
                    type(exc).__name__,
                    (time.monotonic() - started_at) * 1000,
                )
                continue
            LOGGER.info(
                "Model provider %s succeeded (attempt %d, duration_ms=%.0f)",
                model_name,
                attempt,
                (time.monotonic() - started_at) * 1000,
            )
            return response
        raise errors[-1] from errors[0]

    @staticmethod
    def _tool_name(tool: Any) -> str | None:
        if isinstance(tool, dict):
            name = tool.get("name")
            if not name and isinstance(tool.get("function"), dict):
                name = tool["function"].get("name")
            return name if isinstance(name, str) else None
        name = getattr(tool, "name", None)
        return name if isinstance(name, str) else None


def _load_environment() -> None:
    load_dotenv(ENV_FILE, override=False)
    if os.getenv("LANGSMITH_API_KEY"):
        os.environ.setdefault("LANGSMITH_TRACING", "true")
        os.environ.setdefault("LANGSMITH_PROJECT", "getnet-multi-agent")


def build_chat_model() -> tuple[Any, str, list[tuple[str, Callable[[], Any]]]]:
    """Build the configured model and lazy fallback chain."""
    _load_environment()
    settings = load_llm_settings()
    model_id = settings["model"]
    thinking_level = settings.get("thinking_level")
    model = _create_chat_model(model_id, thinking_level)
    fallback_models = [
        (
            fallback,
            lambda fallback=fallback: _create_chat_model(
                fallback,
                thinking_level if fallback.startswith("google_genai:") else None,
            ),
        )
        for fallback in settings["fallback_models"]
    ]
    return model, model_id, fallback_models

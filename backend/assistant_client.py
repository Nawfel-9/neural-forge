"""
assistant_client.py
===================
NVIDIA-hosted OpenAI-compatible chat client for the AI Assistant tab.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from utils.project_state import ProjectState


DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "nvidia/nemotron-mini-4b-instruct"
DEFAULT_TIMEOUT_SECONDS = 20.0
DEFAULT_MAX_TOKENS = 1024
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"


@dataclass(frozen=True)
class AssistantSettings:
    api_key: str
    base_url: str
    model: str
    timeout_seconds: float
    max_tokens: int
    enable_thinking: bool


@dataclass(frozen=True)
class ChatStreamEvent:
    kind: str  # "reasoning" | "content"
    text: str


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        _load_env_file_fallback()
        return
    loaded = load_dotenv(dotenv_path=ENV_PATH, override=False)
    if not loaded:
        _load_env_file_fallback()


def _load_env_file_fallback() -> None:
    """Small .env reader used when python-dotenv is unavailable."""
    if not ENV_PATH.exists():
        return

    for raw_line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def get_assistant_settings() -> AssistantSettings:
    """Read assistant settings from environment variables."""
    _load_dotenv()
    return AssistantSettings(
        api_key=os.getenv("NVIDIA_API_KEY", "").strip(),
        base_url=os.getenv("NVIDIA_BASE_URL", DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL,
        model=os.getenv("NVIDIA_ASSISTANT_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
        timeout_seconds=max(5.0, _env_float("NVIDIA_ASSISTANT_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)),
        max_tokens=max(128, _env_int("NVIDIA_ASSISTANT_MAX_TOKENS", DEFAULT_MAX_TOKENS)),
        enable_thinking=_env_bool("NVIDIA_ASSISTANT_ENABLE_THINKING", False),
    )


def is_assistant_configured() -> bool:
    """Return True when the NVIDIA API key is available."""
    return bool(get_assistant_settings().api_key)


def build_project_context(state: ProjectState) -> str:
    """Build a compact, non-sensitive summary of the current project state."""
    lines: list[str] = []

    if state.dataframe is not None:
        rows, cols = state.dataframe.shape
        lines.append(f"Dataset: {rows} rows x {cols} columns")
        if state.target_column:
            lines.append(f"Target column: {state.target_column}")
        lines.append(f"Problem type: {state.problem_type}")
        lines.append(f"Input features: {state.input_features()}")
        lines.append(f"Output units/classes: {state.output_classes()}")
    else:
        lines.append("Dataset: not loaded")

    split_config = ", ".join(f"{key}={value}" for key, value in state.split_config.items())
    lines.append(f"Split config: {split_config or 'default'}")

    if state.blueprint:
        layer_names = [layer.get("type", "Unknown") for layer in state.blueprint]
        preview = ", ".join(layer_names[:8])
        if len(layer_names) > 8:
            preview += f", ... (+{len(layer_names) - 8} more)"
        lines.append(f"Model blueprint: {len(state.blueprint)} layers ({preview})")
    else:
        lines.append("Model blueprint: empty")

    hyperparams = ", ".join(f"{key}={value}" for key, value in state.hyperparams.items())
    lines.append(f"Training hyperparameters: {hyperparams}")
    lines.append(f"Training device: {state.device}")
    lines.append(f"Loss: {state.loss_fn_name}")
    lines.append(f"Optimizer: {state.optimizer_name}")
    lines.append(f"Preprocessing pipeline: {'ready' if state.pipeline is not None else 'not built'}")
    lines.append(f"PyTorch model: {'built' if state.model is not None else 'not built'}")
    lines.append(f"Export trace input: {'ready' if state.dummy_tensor is not None else 'not ready'}")

    if state.dev_project_path:
        lines.append(f"Developer Mode project path: {state.dev_project_path}")
    else:
        lines.append("Developer Mode project path: not imported")

    return "\n".join(lines)


def stream_chat_response(
    messages: Iterable[dict[str, str]],
    project_context: str = "",
) -> Iterable[ChatStreamEvent]:
    """Stream assistant response text from NVIDIA's OpenAI-compatible endpoint."""
    settings = get_assistant_settings()
    if not settings.api_key:
        raise RuntimeError(
            "NVIDIA_API_KEY is not set. Add it to .env, or copy .env.example to .env "
            "and fill in your NVIDIA API key."
        )

    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The openai package is not installed. Run `pip install -r requirements.txt`."
        ) from exc

    client = OpenAI(
        base_url=settings.base_url,
        api_key=settings.api_key,
        timeout=settings.timeout_seconds,
        max_retries=0,
    )

    system_prompt = (
        "You are Neural Forge's engineering assistant inside a PyQt6 desktop app. "
        "Help the user reason about datasets, model architecture, training settings, "
        "evaluation, export, and Developer Mode projects. Be concise, technical, and "
        "actionable. Do not claim that you changed files or ran training unless the user "
        "explicitly says they did it."
    )

    prepared_messages = [{"role": "system", "content": system_prompt}]
    if project_context:
        prepared_messages.append(
            {
                "role": "system",
                "content": "Current Neural Forge project context:\n" + project_context,
            }
        )
    prepared_messages.extend(list(messages)[-14:])

    completion = client.chat.completions.create(
        model=settings.model,
        messages=prepared_messages,
        temperature=0.7,
        top_p=1,
        max_tokens=settings.max_tokens,
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": settings.enable_thinking,
                "clear_thinking": False,
            }
        },
        stream=True,
    )

    for chunk in completion:
        if not getattr(chunk, "choices", None):
            continue
        if not chunk.choices or getattr(chunk.choices[0], "delta", None) is None:
            continue
        delta = chunk.choices[0].delta
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            yield ChatStreamEvent("reasoning", reasoning)
        content = getattr(delta, "content", None)
        if content:
            yield ChatStreamEvent("content", content)

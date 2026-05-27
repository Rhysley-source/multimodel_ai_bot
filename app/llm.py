"""
LLM service layer — powered by LangChain.
Supports OpenAI, Grok, DeepSeek, and Anthropic through a single unified interface.

Add your keys to .env:
    OPENAI_API_KEY=sk-...
    GROK_API_KEY=xai-...
    DEEPSEEK_API_KEY=sk-...
    ANTHROPIC_API_KEY=sk-ant-...
"""
import logging
from typing import AsyncGenerator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.config import settings

logger = logging.getLogger(__name__)

# Map short model name -> (provider, full_model_id)
MODEL_REGISTRY = {
    # OpenAI
    "gpt":              ("openai",    "gpt-4o"),
    "gpt-mini":         ("openai",    "gpt-4o-mini"),
    # xAI Grok
    "grok":             ("grok",      "grok-3"),
    "grok-mini":        ("grok",      "grok-3-mini"),
    # DeepSeek
    "deepseek":         ("deepseek",  "deepseek-chat"),
    "deepseek-r1":      ("deepseek",  "deepseek-reasoner"),
    # Anthropic Claude
    "claude":           ("anthropic", "claude-sonnet-4-6"),
    "claude-opus":      ("anthropic", "claude-opus-4-7"),
}

DEFAULT_MODEL = settings.DEFAULT_MODEL
LLM_TIMEOUT   = 30.0

# Shared LLM client cache — one instance per (provider, model_id)
# LangChain chat models are stateless and safe to reuse across requests.
_llm_cache: dict[str, object] = {}


class LLMError(Exception):
    """Raised when an LLM call fails or is misconfigured."""

class LLMTimeoutError(LLMError):
    """Provider did not respond within LLM_TIMEOUT seconds."""

class LLMRateLimitError(LLMError):
    """Provider rate-limit / quota exceeded."""

class LLMConnectionError(LLMError):
    """Network connection to the provider failed."""


def resolve_model(short_name: str | None) -> tuple[str, str, str]:
    """Return (short_name, provider, full_model_id)."""
    name = short_name or DEFAULT_MODEL
    if name not in MODEL_REGISTRY:
        raise LLMError(f"Unknown model '{name}'. Available: {', '.join(MODEL_REGISTRY)}")
    provider, full_id = MODEL_REGISTRY[name]
    return name, provider, full_id


def _build_llm(provider: str, model_id: str):
    """Return a cached LangChain chat model — creates once, reuses across requests."""
    cache_key = f"{provider}:{model_id}"
    if cache_key in _llm_cache:
        return _llm_cache[cache_key]

    from langchain_openai import ChatOpenAI

    llm = None

    if provider == "openai":
        if not settings.OPENAI_API_KEY:
            raise LLMError("OPENAI_API_KEY is not set in .env.")
        llm = ChatOpenAI(
            model=model_id,
            api_key=settings.OPENAI_API_KEY,
            max_tokens=1024,
            timeout=LLM_TIMEOUT,
            streaming=True,
        )

    elif provider == "grok":
        if not settings.GROK_API_KEY:
            raise LLMError("GROK_API_KEY is not set in .env.")
        llm = ChatOpenAI(
            model=model_id,
            api_key=settings.GROK_API_KEY,
            base_url="https://api.x.ai/v1",
            max_tokens=1024,
            timeout=LLM_TIMEOUT,
            streaming=True,
        )

    elif provider == "deepseek":
        if not settings.DEEPSEEK_API_KEY:
            raise LLMError("DEEPSEEK_API_KEY is not set in .env.")
        llm = ChatOpenAI(
            model=model_id,
            api_key=settings.DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com/v1",
            max_tokens=1024,
            timeout=LLM_TIMEOUT,
            streaming=True,
        )

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        if not settings.ANTHROPIC_API_KEY:
            raise LLMError("ANTHROPIC_API_KEY is not set in .env.")
        llm = ChatAnthropic(
            model=model_id,
            api_key=settings.ANTHROPIC_API_KEY,
            max_tokens=1024,
            timeout=LLM_TIMEOUT,
        )

    else:
        raise LLMError(f"Unsupported provider: {provider}")

    _llm_cache[cache_key] = llm
    return llm


def _build_messages(history: list[dict]) -> list:
    """Convert history dicts to LangChain message objects with system prompt."""
    msgs = [SystemMessage(content=settings.SYSTEM_PROMPT)]
    for m in history:
        if m["role"] == "user":
            msgs.append(HumanMessage(content=m["content"]))
        else:
            msgs.append(AIMessage(content=m["content"]))
    return msgs


def _wrap_error(e: Exception) -> LLMError:
    """Map provider-specific exceptions to our LLMError hierarchy."""
    msg = str(e).lower()
    if "timeout" in msg or "timed out" in msg:
        return LLMTimeoutError("The AI took too long to respond. Please try again.")
    if "rate" in msg or "quota" in msg or "429" in msg:
        return LLMRateLimitError("Too many requests. Please wait a moment and try again.")
    if "connection" in msg or "network" in msg:
        return LLMConnectionError("Could not connect to the AI service. Please try again.")
    if "401" in msg or "invalid" in msg or "unauthorized" in msg or "ip" in msg:
        return LLMError(f"The AI service returned an error (HTTP 401). Please try again.")
    if "status" in msg or "http" in msg:
        return LLMError(f"The AI service returned an error. Please try again.")
    logger.exception("Unexpected LLM error: %s", e)
    return LLMError("Unexpected error from AI provider. Please try again.")


async def generate_reply(short_name: str | None, history: list[dict]) -> tuple[str, str]:
    """
    Send history to the chosen model and return (reply_text, model_name).
    history: list of {"role": "user"|"assistant", "content": "..."}, oldest first.
    """
    name, provider, model_id = resolve_model(short_name)
    llm = _build_llm(provider, model_id)
    messages = _build_messages(history)

    logger.info("LLM request | model=%s provider=%s messages=%d", name, provider, len(messages))
    try:
        response = await llm.ainvoke(messages)
        reply = response.content
    except LLMError:
        raise
    except Exception as e:
        raise _wrap_error(e) from e

    logger.info("LLM response | model=%s chars=%d", name, len(reply))
    return reply, name


async def stream_reply(short_name: str | None, history: list[dict]) -> AsyncGenerator[str, None]:
    """
    Async generator yielding text chunks from the chosen model.
    Raises an LLMError subclass on failure.
    """
    name, provider, model_id = resolve_model(short_name)
    llm = _build_llm(provider, model_id)
    messages = _build_messages(history)

    logger.info("LLM stream | model=%s provider=%s messages=%d", name, provider, len(messages))
    try:
        async for chunk in llm.astream(messages):
            if chunk.content:
                yield chunk.content
    except LLMError:
        raise
    except Exception as e:
        raise _wrap_error(e) from e

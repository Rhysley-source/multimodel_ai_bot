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

# Identity prefix injected into the system prompt so each model knows who it is.
# DeepSeek models especially tend to claim they are Claude without this.
MODEL_IDENTITY = {
    "gpt":          "You are GPT-4o, a large language model made by OpenAI.",
    "gpt-mini":     "You are GPT-4o mini, a large language model made by OpenAI.",
    "grok":         "You are Grok 3, an AI assistant made by xAI.",
    "grok-mini":    "You are Grok 3 mini, an AI assistant made by xAI.",
    "deepseek":     "You are DeepSeek Chat, an AI assistant made by DeepSeek.",
    "deepseek-r1":  "You are DeepSeek R1, a reasoning AI assistant made by DeepSeek.",
    "claude":       "You are Claude Sonnet, an AI assistant made by Anthropic.",
    "claude-opus":  "You are Claude Opus, an AI assistant made by Anthropic.",
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
        logger.warning("Model resolve failed | requested=%s available=%s", name, list(MODEL_REGISTRY))
        raise LLMError(f"Unknown model '{name}'. Available: {', '.join(MODEL_REGISTRY)}")
    provider, full_id = MODEL_REGISTRY[name]
    logger.debug("Model resolved | short=%s provider=%s model_id=%s", name, provider, full_id)
    return name, provider, full_id


def _build_llm(provider: str, model_id: str):
    """Return a cached LangChain chat model — creates once, reuses across requests."""
    cache_key = f"{provider}:{model_id}"
    if cache_key in _llm_cache:
        logger.debug("LLM client cache hit | %s", cache_key)
        return _llm_cache[cache_key]

    logger.info("LLM client init | provider=%s model_id=%s", provider, model_id)
    from langchain_openai import ChatOpenAI

    llm = None

    if provider == "openai":
        if not settings.OPENAI_API_KEY:
            logger.error("LLM client init failed | provider=openai reason=OPENAI_API_KEY not set")
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
            logger.error("LLM client init failed | provider=grok reason=GROK_API_KEY not set")
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
            logger.error("LLM client init failed | provider=deepseek reason=DEEPSEEK_API_KEY not set")
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
            logger.error("LLM client init failed | provider=anthropic reason=ANTHROPIC_API_KEY not set")
            raise LLMError("ANTHROPIC_API_KEY is not set in .env.")
        llm = ChatAnthropic(
            model=model_id,
            api_key=settings.ANTHROPIC_API_KEY,
            max_tokens=1024,
            timeout=LLM_TIMEOUT,
        )

    else:
        logger.error("LLM client init failed | provider=%s reason=unsupported", provider)
        raise LLMError(f"Unsupported provider: {provider}")

    _llm_cache[cache_key] = llm
    logger.info("LLM client ready | provider=%s model_id=%s", provider, model_id)
    return llm


def _build_messages(history: list[dict], model_name: str) -> list:
    """Convert history dicts to LangChain message objects with system prompt."""
    identity = MODEL_IDENTITY.get(model_name, "")
    system_content = f"{identity}\n\n{settings.SYSTEM_PROMPT}" if identity else settings.SYSTEM_PROMPT
    msgs = [SystemMessage(content=system_content)]
    for m in history:
        if m["role"] == "user":
            msgs.append(HumanMessage(content=m["content"]))
        else:
            msgs.append(AIMessage(content=m["content"]))
    logger.debug("Messages built | model=%s total=%d (1 system + %d history)", model_name, len(msgs), len(history))
    return msgs


def _wrap_error(e: Exception) -> LLMError:
    """Map provider-specific exceptions to our LLMError hierarchy."""
    msg = str(e).lower()
    if "timeout" in msg or "timed out" in msg:
        return LLMTimeoutError("The AI took too long to respond. Please try again.")
    if "rate" in msg or "quota" in msg or "429" in msg:
        return LLMRateLimitError("Too many requests. Please wait a moment and try again.")
    if "credits" in msg or "spending" in msg or "billing" in msg or "balance" in msg:
        return LLMRateLimitError("AI provider credits exhausted. Please top up your account.")
    if "403" in msg or "permission" in msg:
        return LLMError("Access denied by AI provider. Check your API key permissions.")
    if "connection" in msg or "network" in msg:
        return LLMConnectionError("Could not connect to the AI service. Please try again.")
    if "401" in msg or "invalid" in msg or "unauthorized" in msg or "ip" in msg:
        return LLMError("The AI service returned an auth error (HTTP 401). Please try again.")
    if "status" in msg or "http" in msg:
        return LLMError("The AI service returned an error. Please try again.")
    logger.exception("Unexpected LLM error: %s", e)
    return LLMError("Unexpected error from AI provider. Please try again.")


async def generate_reply(short_name: str | None, history: list[dict]) -> tuple[str, str]:
    """
    Send history to the chosen model and return (reply_text, model_name).
    history: list of {"role": "user"|"assistant", "content": "..."}, oldest first.
    """
    name, provider, model_id = resolve_model(short_name)
    llm = _build_llm(provider, model_id)
    messages = _build_messages(history, name)

    logger.info("LLM request | model=%s provider=%s model_id=%s messages=%d",
                name, provider, model_id, len(messages))
    try:
        import time
        t0 = time.monotonic()
        response = await llm.ainvoke(messages)
        elapsed = time.monotonic() - t0
        reply = response.content
    except LLMError:
        raise
    except Exception as e:
        logger.error("LLM request failed | model=%s provider=%s error=%s", name, provider, e)
        raise _wrap_error(e) from e

    logger.info(
        "LLM response OK | model=%s provider=%s model_id=%s chars=%d elapsed=%.2fs",
        name, provider, model_id, len(reply), elapsed,
    )
    return reply, name


async def stream_reply(short_name: str | None, history: list[dict]) -> AsyncGenerator[str, None]:
    """
    Async generator yielding text chunks from the chosen model.
    Raises an LLMError subclass on failure.
    """
    name, provider, model_id = resolve_model(short_name)
    llm = _build_llm(provider, model_id)
    messages = _build_messages(history, name)

    logger.info("LLM stream start | model=%s provider=%s model_id=%s messages=%d",
                name, provider, model_id, len(messages))
    try:
        import time
        t0 = time.monotonic()
        chunk_count = 0
        total_chars = 0
        async for chunk in llm.astream(messages):
            if chunk.content:
                chunk_count += 1
                total_chars += len(chunk.content)
                yield chunk.content
    except LLMError:
        raise
    except Exception as e:
        logger.error("LLM stream failed | model=%s provider=%s error=%s", name, provider, e)
        raise _wrap_error(e) from e

    elapsed = time.monotonic() - t0
    logger.info(
        "LLM stream OK | model=%s provider=%s model_id=%s chunks=%d chars=%d elapsed=%.2fs",
        name, provider, model_id, chunk_count, total_chars, elapsed,
    )

"""
LLM service layer. Abstracts over multiple providers (Anthropic, OpenAI)
so routes don't depend on a specific SDK.

Add your keys to .env:
    ANTHROPIC_API_KEY=sk-ant-...
    OPENAI_API_KEY=sk-...
"""
import asyncio

from app.config import settings

# Map a short model name -> (provider, full_model_id)
MODEL_REGISTRY = {
    "claude": ("anthropic", "claude-sonnet-4-6"),
    "claude-opus": ("anthropic", "claude-opus-4-7"),
    "gpt": ("openai", "gpt-4o"),
    "gpt-mini": ("openai", "gpt-4o-mini"),
}

DEFAULT_MODEL = settings.DEFAULT_MODEL
LLM_TIMEOUT = 30.0  # seconds


class LLMError(Exception):
    """Raised when an LLM call fails or is misconfigured."""


class LLMTimeoutError(LLMError):
    """Raised when the provider does not respond within LLM_TIMEOUT seconds."""


class LLMRateLimitError(LLMError):
    """Raised when the provider returns a rate-limit / quota error."""


class LLMConnectionError(LLMError):
    """Raised when the network connection to the provider fails."""


def resolve_model(short_name: str | None) -> tuple[str, str, str]:
    """Return (short_name, provider, full_model_id) for a requested model."""
    name = short_name or DEFAULT_MODEL
    if name not in MODEL_REGISTRY:
        raise LLMError(
            f"Unknown model '{name}'. Available: {', '.join(MODEL_REGISTRY)}"
        )
    provider, full_id = MODEL_REGISTRY[name]
    return name, provider, full_id


async def _call_anthropic(model_id: str, history: list[dict]) -> str:
    import anthropic

    key = settings.ANTHROPIC_API_KEY
    if not key:
        raise LLMError("The AI service is not configured. Please contact support.")

    client = anthropic.AsyncAnthropic(api_key=key)
    try:
        resp = await asyncio.wait_for(
            client.messages.create(
                model=model_id,
                max_tokens=1024,
                messages=history,
            ),
            timeout=LLM_TIMEOUT,
        )
    except asyncio.TimeoutError:
        raise LLMTimeoutError("The AI took too long to respond. Please try again.")
    except anthropic.RateLimitError:
        raise LLMRateLimitError("Too many requests. Please wait a moment and try again.")
    except anthropic.APIConnectionError:
        raise LLMConnectionError("Could not connect to the AI service. Please try again.")
    except anthropic.APIStatusError as e:
        raise LLMError(f"The AI service returned an error (HTTP {e.status_code}). Please try again.")
    except Exception as e:
        raise LLMError(f"Unexpected error from AI provider. Please try again.")

    return "".join(
        block.text for block in resp.content if getattr(block, "type", None) == "text"
    )


async def _call_openai(model_id: str, history: list[dict]) -> str:
    from openai import AsyncOpenAI, APITimeoutError, RateLimitError, APIConnectionError, APIStatusError

    key = settings.OPENAI_API_KEY
    if not key:
        raise LLMError("The AI service is not configured. Please contact support.")

    client = AsyncOpenAI(api_key=key, timeout=LLM_TIMEOUT)
    try:
        resp = await client.chat.completions.create(
            model=model_id,
            max_tokens=1024,
            messages=history,
        )
    except APITimeoutError:
        raise LLMTimeoutError("The AI took too long to respond. Please try again.")
    except RateLimitError:
        raise LLMRateLimitError("Too many requests. Please wait a moment and try again.")
    except APIConnectionError:
        raise LLMConnectionError("Could not connect to the AI service. Please try again.")
    except APIStatusError as e:
        raise LLMError(f"The AI service returned an error (HTTP {e.status_code}). Please try again.")
    except Exception:
        raise LLMError("Unexpected error from AI provider. Please try again.")

    return resp.choices[0].message.content or ""


async def generate_reply(short_name: str | None, history: list[dict]) -> tuple[str, str]:
    """
    Send conversation `history` to the chosen model and return (reply_text, model_name).
    `history` is a list of {"role", "content"} dicts, oldest first, ending with
    the latest user message.
    """
    name, provider, model_id = resolve_model(short_name)

    if provider == "anthropic":
        reply = await _call_anthropic(model_id, history)
    elif provider == "openai":
        reply = await _call_openai(model_id, history)
    else:
        raise LLMError(f"Unsupported provider: {provider}")

    return reply, name


# ---------- Streaming variants ----------

async def _stream_anthropic(model_id: str, history: list[dict]):
    import anthropic

    key = settings.ANTHROPIC_API_KEY
    if not key:
        raise LLMError("The AI service is not configured. Please contact support.")

    client = anthropic.AsyncAnthropic(api_key=key)
    try:
        async with client.messages.stream(
            model=model_id,
            max_tokens=1024,
            messages=history,
        ) as stream:
            async for text in stream.text_stream:
                yield text
    except anthropic.RateLimitError:
        raise LLMRateLimitError("Too many requests. Please wait a moment and try again.")
    except anthropic.APIConnectionError:
        raise LLMConnectionError("Could not connect to the AI service. Please try again.")
    except anthropic.APIStatusError as e:
        raise LLMError(f"The AI service returned an error (HTTP {e.status_code}). Please try again.")
    except LLMError:
        raise
    except Exception:
        raise LLMError("Unexpected error from AI provider. Please try again.")


async def _stream_openai(model_id: str, history: list[dict]):
    from openai import AsyncOpenAI, APITimeoutError, RateLimitError, APIConnectionError, APIStatusError

    key = settings.OPENAI_API_KEY
    if not key:
        raise LLMError("The AI service is not configured. Please contact support.")

    client = AsyncOpenAI(api_key=key, timeout=LLM_TIMEOUT)
    try:
        stream = await client.chat.completions.create(
            model=model_id,
            max_tokens=1024,
            messages=history,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
    except APITimeoutError:
        raise LLMTimeoutError("The AI took too long to respond. Please try again.")
    except RateLimitError:
        raise LLMRateLimitError("Too many requests. Please wait a moment and try again.")
    except APIConnectionError:
        raise LLMConnectionError("Could not connect to the AI service. Please try again.")
    except APIStatusError as e:
        raise LLMError(f"The AI service returned an error (HTTP {e.status_code}). Please try again.")
    except LLMError:
        raise
    except Exception:
        raise LLMError("Unexpected error from AI provider. Please try again.")


async def stream_reply(short_name: str | None, history: list[dict]):
    """
    Async generator yielding text chunks from the chosen model.
    Raises an LLMError subclass on failure — caller sends it to the client.
    """
    name, provider, model_id = resolve_model(short_name)

    if provider == "anthropic":
        gen = _stream_anthropic(model_id, history)
    elif provider == "openai":
        gen = _stream_openai(model_id, history)
    else:
        raise LLMError(f"Unsupported provider: {provider}")

    async for chunk in gen:
        yield chunk

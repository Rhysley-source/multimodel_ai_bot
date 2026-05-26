"""
LLM service layer. Abstracts over multiple providers (Anthropic, OpenAI)
so routes don't depend on a specific SDK.

Add your keys to .env:
    ANTHROPIC_API_KEY=sk-ant-...
    OPENAI_API_KEY=sk-...
"""
import os

# Map a short model name -> (provider, full_model_id)
MODEL_REGISTRY = {
    "claude": ("anthropic", "claude-sonnet-4-20250514"),
    "claude-opus": ("anthropic", "claude-opus-4-20250514"),
    "gpt": ("openai", "gpt-4o"),
    "gpt-mini": ("openai", "gpt-4o-mini"),
}

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "claude")


class LLMError(Exception):
    """Raised when an LLM call fails or is misconfigured."""


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

    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise LLMError("ANTHROPIC_API_KEY is not set in the environment.")

    client = anthropic.AsyncAnthropic(api_key=key)
    resp = await client.messages.create(
        model=model_id,
        max_tokens=1024,
        messages=history,  # [{"role": "user"|"assistant", "content": "..."}]
    )
    # Concatenate any text blocks in the response.
    return "".join(
        block.text for block in resp.content if getattr(block, "type", None) == "text"
    )


async def _call_openai(model_id: str, history: list[dict]) -> str:
    from openai import AsyncOpenAI

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise LLMError("OPENAI_API_KEY is not set in the environment.")

    client = AsyncOpenAI(api_key=key)
    resp = await client.chat.completions.create(
        model=model_id,
        max_tokens=1024,
        messages=history,
    )
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

    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise LLMError("ANTHROPIC_API_KEY is not set in the environment.")

    client = anthropic.AsyncAnthropic(api_key=key)
    async with client.messages.stream(
        model=model_id,
        max_tokens=1024,
        messages=history,
    ) as stream:
        async for text in stream.text_stream:
            yield text


async def _stream_openai(model_id: str, history: list[dict]):
    from openai import AsyncOpenAI

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise LLMError("OPENAI_API_KEY is not set in the environment.")

    client = AsyncOpenAI(api_key=key)
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


async def stream_reply(short_name: str | None, history: list[dict]):
    """
    Async generator yielding text chunks from the chosen model.
    Yields a final tuple sentinel? No — caller accumulates chunks itself.
    Raises LLMError before the first yield if misconfigured.
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
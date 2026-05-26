import json
import logging

from fastapi import APIRouter, Depends, HTTPException, status

logger = logging.getLogger(__name__)
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import settings
from app.database import AsyncSessionLocal, get_db
from app.dependencies import get_current_user
from app.llm import (
    LLMConnectionError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
    generate_reply,
    resolve_model,
    stream_reply,
)
from app.models import Conversation, Message, User
from app.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationDetail,
    ConversationOut,
)


def _llm_error_response(e: LLMError, user_message: str) -> JSONResponse:
    """Map an LLMError to the right HTTP status and echo back the user's message."""
    if isinstance(e, LLMTimeoutError):
        code = status.HTTP_504_GATEWAY_TIMEOUT
    elif isinstance(e, LLMRateLimitError):
        code = status.HTTP_429_TOO_MANY_REQUESTS
    elif isinstance(e, LLMConnectionError):
        code = status.HTTP_503_SERVICE_UNAVAILABLE
    else:
        code = status.HTTP_502_BAD_GATEWAY
    return JSONResponse(
        status_code=code,
        content={"detail": str(e), "user_message": user_message},
    )

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/conversations", response_model=ConversationOut, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Create a new empty conversation and return its id."""
    conversation = Conversation(user_id=current_user.id)
    db.add(conversation)
    await db.commit()
    await db.refresh(conversation)
    return conversation


@router.post("", response_model=ChatResponse)
async def send_message(
    payload: ChatRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Send a user query, get an LLM reply. Persists both messages."""
    # 1. Get or create the conversation (and verify ownership).
    existing_messages = []
    if payload.conversation_id is not None:
        result = await db.execute(
            select(Conversation)
            .where(Conversation.id == payload.conversation_id)
            .options(selectinload(Conversation.messages))
        )
        conversation = result.scalar_one_or_none()
        if conversation is None or conversation.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found",
            )
        existing_messages = conversation.messages  # eagerly loaded above
    else:
        conversation = Conversation(
            user_id=current_user.id,
            title=payload.message[:60],  # first message as a rough title
        )
        db.add(conversation)
        await db.flush()  # assign an id without committing yet

    # 2. Build the history — keep only the most recent HISTORY_LIMIT messages.
    recent = existing_messages[-settings.HISTORY_LIMIT:]
    history = [{"role": m.role, "content": m.content} for m in recent]
    history.append({"role": "user", "content": payload.message})

    # 3. Call the LLM.
    try:
        reply_text, model_name = await generate_reply(payload.model, history)
    except LLMError as e:
        return _llm_error_response(e, payload.message)

    # 4. Persist both the user message and the assistant reply.
    db.add(Message(
        conversation_id=conversation.id, role="user", content=payload.message
    ))
    db.add(Message(
        conversation_id=conversation.id, role="assistant",
        content=reply_text, model=model_name,
    ))
    await db.commit()

    return ChatResponse(
        conversation_id=conversation.id,
        reply=reply_text,
        model=model_name,
    )


@router.post("/stream")
async def send_message_stream(
    payload: ChatRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Stream an LLM reply token-by-token using Server-Sent Events (SSE).

    Event stream format (each line is `data: <json>\\n\\n`):
      {"type": "meta",  "conversation_id": 12, "model": "claude"}
      {"type": "chunk", "delta": "Hello"}
      {"type": "chunk", "delta": " there"}
      {"type": "done",  "conversation_id": 12}
      {"type": "error", "detail": "..."}   (on failure)
    """
    # Validate the model up front so a bad model is a clean HTTP 400, not mid-stream.
    try:
        model_name, _provider, _model_id = resolve_model(payload.model)
    except LLMError as e:
        logger.warning("Stream rejected — invalid model: %s", payload.model)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    existing_messages = []
    if payload.conversation_id is not None:
        result = await db.execute(
            select(Conversation)
            .where(Conversation.id == payload.conversation_id)
            .options(selectinload(Conversation.messages))
        )
        conversation = result.scalar_one_or_none()
        if conversation is None or conversation.user_id != current_user.id:
            logger.warning(
                "Stream rejected — conversation not found | user_id=%s conv_id=%s",
                current_user.id, payload.conversation_id,
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found",
            )
        existing_messages = conversation.messages
    else:
        conversation = Conversation(
            user_id=current_user.id, title=payload.message[:60]
        )
        db.add(conversation)
        await db.flush()

    conversation_id = conversation.id
    user_message = payload.message

    # Keep only the most recent HISTORY_LIMIT messages for LLM context.
    recent = existing_messages[-settings.HISTORY_LIMIT:]
    history = [{"role": m.role, "content": m.content} for m in recent]
    history.append({"role": "user", "content": user_message})

    await db.commit()
    logger.info(
        "Stream start | user_id=%s conv_id=%s model=%s "
        "total_msgs=%d sent_to_llm=%d",
        current_user.id, conversation_id, model_name,
        len(existing_messages), len(history),
    )

    async def event_generator():
        def sse(obj: dict) -> str:
            return f"data: {json.dumps(obj)}\n\n"

        yield sse({"type": "meta", "conversation_id": conversation_id, "model": model_name})

        full_reply = []
        try:
            async for delta in stream_reply(payload.model, history):
                full_reply.append(delta)
                yield sse({"type": "chunk", "delta": delta})
        except LLMTimeoutError as e:
            logger.warning("Stream timeout | conv_id=%s model=%s", conversation_id, model_name)
            yield sse({"type": "error", "code": 504, "detail": str(e), "user_message": user_message})
            return
        except LLMRateLimitError as e:
            logger.warning("Stream rate-limit | conv_id=%s model=%s", conversation_id, model_name)
            yield sse({"type": "error", "code": 429, "detail": str(e), "user_message": user_message})
            return
        except LLMConnectionError as e:
            logger.warning("Stream connection error | conv_id=%s model=%s", conversation_id, model_name)
            yield sse({"type": "error", "code": 503, "detail": str(e), "user_message": user_message})
            return
        except LLMError as e:
            logger.error("Stream LLM error | conv_id=%s model=%s error=%s", conversation_id, model_name, e)
            yield sse({"type": "error", "code": 502, "detail": str(e), "user_message": user_message})
            return
        except Exception as e:
            logger.exception("Stream unexpected error | conv_id=%s model=%s", conversation_id, model_name)
            yield sse({"type": "error", "code": 502, "detail": "Something went wrong. Please try again.", "user_message": user_message})
            return

        reply_text = "".join(full_reply)
        logger.info(
            "Stream done | conv_id=%s model=%s chunks=%d chars=%d",
            conversation_id, model_name, len(full_reply), len(reply_text),
        )

        async with AsyncSessionLocal() as session:
            session.add(Message(
                conversation_id=conversation_id, role="user", content=user_message
            ))
            session.add(Message(
                conversation_id=conversation_id, role="assistant",
                content=reply_text, model=model_name,
            ))
            await session.commit()
            logger.info("Stream messages saved | conv_id=%s", conversation_id)

        yield sse({"type": "done", "conversation_id": conversation_id})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable nginx buffering if proxied
        },
    )


@router.get("/conversations", response_model=list[ConversationOut])
async def list_conversations(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List the current user's conversations, newest first."""
    result = await db.execute(
        select(Conversation)
        .where(Conversation.user_id == current_user.id)
        .order_by(Conversation.created_at.desc())
    )
    return result.scalars().all()


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(
    conversation_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get one conversation with its full message history."""
    result = await db.execute(
        select(Conversation)
        .where(Conversation.id == conversation_id)
        .options(selectinload(Conversation.messages))
    )
    conversation = result.scalar_one_or_none()
    if conversation is None or conversation.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found"
        )
    return conversation
import json

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import AsyncSessionLocal, get_db
from app.dependencies import get_current_user
from app.llm import LLMError, generate_reply, resolve_model, stream_reply
from app.models import Conversation, Message, User
from app.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationDetail,
    ConversationOut,
)

router = APIRouter(prefix="/chat", tags=["chat"])


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

    # 2. Build the history to send to the model (oldest first).
    history = [
        {"role": m.role, "content": m.content} for m in existing_messages
    ]
    history.append({"role": "user", "content": payload.message})

    # 3. Call the LLM.
    try:
        reply_text, model_name = await generate_reply(payload.model, history)
    except LLMError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e)
        )

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
    # --- Pre-stream work, using the request-scoped session ---
    # Validate the model up front so a bad model is a clean HTTP 400, not mid-stream.
    try:
        model_name, _provider, _model_id = resolve_model(payload.model)
    except LLMError as e:
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

    history = [{"role": m.role, "content": m.content} for m in existing_messages]
    history.append({"role": "user", "content": user_message})

    # Commit the conversation (and, for a new one, make it durable) before streaming.
    await db.commit()

    async def event_generator():
        def sse(obj: dict) -> str:
            return f"data: {json.dumps(obj)}\n\n"

        # First event: hand the client the conversation id + model.
        yield sse({"type": "meta", "conversation_id": conversation_id,
                   "model": model_name})

        full_reply = []
        try:
            async for delta in stream_reply(payload.model, history):
                full_reply.append(delta)
                yield sse({"type": "chunk", "delta": delta})
        except LLMError as e:
            yield sse({"type": "error", "detail": str(e)})
            return
        except Exception as e:  # surface unexpected provider errors to the client
            yield sse({"type": "error", "detail": f"Stream failed: {e}"})
            return

        reply_text = "".join(full_reply)

        # Persist both messages using a FRESH session (the request session is gone).
        async with AsyncSessionLocal() as session:
            session.add(Message(
                conversation_id=conversation_id, role="user", content=user_message
            ))
            session.add(Message(
                conversation_id=conversation_id, role="assistant",
                content=reply_text, model=model_name,
            ))
            await session.commit()

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
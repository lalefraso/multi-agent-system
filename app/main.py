from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from threading import BoundedSemaphore
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, status
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(ENV_FILE, override=False)

from app.auth import AuthenticatedUser, require_authenticated_user
from app.agents.router import build_router_agent
from app.customer_data import CustomerDataError
from app.conversation_memory import (
    ConversationBusyError,
    ConversationMemoryError,
    build_conversation_history,
)
from app.llm import LLMConfigurationError
from app.schemas import (
    AgentName,
    AgentStatus,
    ChatRequest,
    ChatResponse,
    ErrorResponse,
)


def _positive_int_setting(name: str, default: int) -> int:
    value = os.getenv(name)
    try:
        parsed = default if value is None else int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if parsed < 1:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


MAX_CONCURRENT_CHATS = _positive_int_setting(
    "CHAT_MAX_CONCURRENT_REQUESTS", 8
)
_chat_slots = BoundedSemaphore(MAX_CONCURRENT_CHATS)


app = FastAPI(title="Getnet Multi-Agent API", version="0.1.0")


def _router_agent(user_id: str):
    try:
        return build_router_agent(user_id)
    except LLMConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


def _message_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(block["text"])
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        )
    return ""


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def chat(
    request: ChatRequest,
    authenticated_user: AuthenticatedUser = Depends(require_authenticated_user),
) -> ChatResponse:
    if not _chat_slots.acquire(blocking=False):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="The service is handling its maximum number of chat requests.",
            headers={"Retry-After": "1"},
        )
    try:
        return _process_chat(request, authenticated_user.user_id)
    finally:
        _chat_slots.release()


def _process_chat(request: ChatRequest, user_id: str) -> ChatResponse:
    conversation_id = request.conversation_id or uuid4()
    user_message = request.message
    try:
        router_agent = _router_agent(user_id)
        history = build_conversation_history(user_id, conversation_id)
    except LLMConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except ConversationMemoryError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    try:
        try:
            history.acquire_turn_lock()
            messages = history.load_context()
            messages.append(HumanMessage(content=request.message))
            result = router_agent.invoke({"messages": messages})

            result_messages = result.get("messages", [])
            agents_used: list[AgentName] = []
            customer_support_output: str | None = None

            for message in result_messages:
                if not isinstance(message, ToolMessage):
                    continue
                if message.name == "knowledge_agent":
                    agents_used.append(AgentName.KNOWLEDGE)
                elif message.name == "customer_support_agent":
                    agents_used.append(AgentName.CUSTOMER_SUPPORT)
                    customer_support_output = _message_text(message.content)

            if (
                AgentName.KNOWLEDGE not in agents_used
                and customer_support_output is not None
            ):
                answer = customer_support_output
            else:
                answer = next(
                    (
                        _message_text(message.content)
                        for message in reversed(result_messages)
                        if isinstance(message, AIMessage)
                        and _message_text(message.content)
                    ),
                    "",
                )

            history.save_turn(user_message, answer)
        finally:
            try:
                history.release_turn_lock()
            finally:
                history.close()
    except ConversationBusyError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
            headers={"Retry-After": "1"},
        ) from exc
    except ConversationMemoryError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except CustomerDataError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    return ChatResponse(
        conversation_id=conversation_id,
        status=AgentStatus.COMPLETED,
        answer=answer,
        agents_used=list(dict.fromkeys(agents_used)),
    )

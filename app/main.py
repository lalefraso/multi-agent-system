from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException, status
from langchain_core.messages import AIMessage, ToolMessage

from app.agents.router import build_router_agent
from app.customer_data import CustomerDataError
from app.llm import LLMConfigurationError
from app.schemas import (
    AgentName,
    AgentStatus,
    ChatRequest,
    ChatResponse,
    ErrorResponse,
)


app = FastAPI(title="Getnet Multi-Agent API", version="0.1.0")


@lru_cache(maxsize=1)
def _router_agent():
    try:
        return build_router_agent()
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
    responses={503: {"model": ErrorResponse}},
)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        result = _router_agent().invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": request.model_dump_json(),
                    }
                ]
            }
        )
    except LLMConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except CustomerDataError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    messages = result.get("messages", [])
    agents_used: list[AgentName] = []
    customer_support_output: str | None = None

    for message in messages:
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
                for message in reversed(messages)
                if isinstance(message, AIMessage) and _message_text(message.content)
            ),
            "",
        )
    return ChatResponse(
        status=AgentStatus.COMPLETED,
        answer=answer,
        agents_used=list(dict.fromkeys(agents_used)),
    )

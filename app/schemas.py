from __future__ import annotations

from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AgentName(StrEnum):
    KNOWLEDGE = "knowledge"
    CUSTOMER_SUPPORT = "customer_support"


class AgentStatus(StrEnum):
    COMPLETED = "completed"
    NOT_IMPLEMENTED = "not_implemented"


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=4000)
    user_id: str = Field(min_length=1, max_length=128)

    @field_validator("message", "user_id")
    @classmethod
    def reject_blank_values(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class ChatResponse(BaseModel):
    request_id: UUID = Field(default_factory=uuid4)
    status: AgentStatus
    answer: str
    agents_used: list[AgentName] = Field(default_factory=list)


class AgentToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=4000)
    user_id: str = Field(min_length=1, max_length=128)


class KnowledgeAgentInput(AgentToolInput):
    pass


class CustomerSupportAgentInput(AgentToolInput):
    pass


class AgentResult(BaseModel):
    agent: AgentName
    status: AgentStatus
    message: str


class ErrorResponse(BaseModel):
    detail: str

from __future__ import annotations

from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AgentName(StrEnum):
    KNOWLEDGE = "knowledge"
    CUSTOMER_SUPPORT = "customer_support"


class AgentStatus(StrEnum):
    COMPLETED = "completed"


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=4000)
    user_id: str = Field(min_length=1, max_length=128)
    conversation_id: UUID | None = None

    @field_validator("message", "user_id")
    @classmethod
    def reject_blank_values(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class ChatResponse(BaseModel):
    request_id: UUID = Field(default_factory=uuid4)
    conversation_id: UUID
    status: AgentStatus
    answer: str
    agents_used: list[AgentName] = Field(default_factory=list)


class AgentToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=4000)
    user_id: str = Field(min_length=1, max_length=128)


class KnowledgeAgentInput(AgentToolInput):
    pass


class CustomerSupportAgentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(min_length=1, max_length=128)

    @field_validator("user_id")
    @classmethod
    def reject_blank_user_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class VectorSearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=4000)
    result_count: int | None = Field(default=None, ge=1, le=10)


class ExternalResourceSearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=4000)


class AgentResult(BaseModel):
    agent: AgentName
    status: AgentStatus
    message: str


class ErrorResponse(BaseModel):
    detail: str

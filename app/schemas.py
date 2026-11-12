from __future__ import annotations

from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AgentName(StrEnum):
    KNOWLEDGE = "knowledge"
    CUSTOMER_SUPPORT = "customer_support"
    HUMAN_ESCALATION = "human_escalation"


class AgentStatus(StrEnum):
    COMPLETED = "completed"
    ESCALATED = "escalated"


class EscalationReason(StrEnum):
    HUMAN_REQUESTED = "human_requested"
    ACTION_UNAVAILABLE = "action_unavailable"
    RISK_OR_COMPLAINT = "risk_or_complaint"
    UNRESOLVED = "unresolved"


class EscalationStatus(StrEnum):
    PENDING = "pending"
    TAKEN = "taken"
    CLOSED = "closed"


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=4000)
    customer_id: str | None = Field(default=None, min_length=1, max_length=128)
    conversation_id: UUID | None = None

    @field_validator("message")
    @classmethod
    def reject_blank_values(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("customer_id")
    @classmethod
    def reject_blank_customer_id(cls, value: str | None) -> str | None:
        if value is not None and (not value.strip() or value != value.strip()):
            raise ValueError("must not be blank or contain surrounding whitespace")
        return value


class ChatResponse(BaseModel):
    request_id: UUID = Field(default_factory=uuid4)
    conversation_id: UUID
    status: AgentStatus
    answer: str
    agents_used: list[AgentName] = Field(default_factory=list)
    escalation_id: UUID | None = None


class AgentToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=4000)


class KnowledgeAgentInput(AgentToolInput):
    pass


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

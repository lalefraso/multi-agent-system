from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from redis import Redis
from redis.exceptions import RedisError

from app.schemas import EscalationReason, EscalationStatus


ESCALATION_KEY_PREFIX = "getnet:escalation:conversation:"


class EscalationError(RuntimeError):
    """Raised when an escalation cannot be safely retrieved or stored."""


class EscalationOwnershipError(EscalationError):
    """Raised when the conversation belongs to another identity."""


class EscalationDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    summary: str = Field(min_length=1, max_length=4000)


class EscalationProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: EscalationReason
    draft: EscalationDraft


class Escalation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    escalation_id: UUID
    conversation_id: UUID
    owner_token: str
    customer_id: str
    status: EscalationStatus
    reason: EscalationReason
    summary: str = Field(min_length=1, max_length=4000)
    created_at: datetime


def get_escalation(
    client: Redis, conversation_id: UUID, owner_token: str
) -> Escalation | None:
    try:
        value = client.get(f"{ESCALATION_KEY_PREFIX}{conversation_id}")
        if value is None:
            return None
        escalation = Escalation.model_validate_json(value)
    except (RedisError, ValidationError, ValueError) as exc:
        raise EscalationError("Could not retrieve escalation") from exc
    if escalation.conversation_id != conversation_id:
        raise EscalationError("Invalid escalation conversation")
    if escalation.owner_token != owner_token:
        raise EscalationOwnershipError("Conversation not found")
    return escalation


def create_escalation(
    client: Redis,
    conversation_id: UUID,
    owner_token: str,
    customer_id: str,
    reason: EscalationReason,
    draft: EscalationDraft,
) -> Escalation:
    escalation = Escalation(
        escalation_id=uuid4(),
        conversation_id=conversation_id,
        owner_token=owner_token,
        customer_id=customer_id,
        status=EscalationStatus.PENDING,
        reason=reason,
        summary=draft.summary,
        created_at=datetime.now(timezone.utc),
    )
    try:
        created = client.set(
            f"{ESCALATION_KEY_PREFIX}{conversation_id}",
            escalation.model_dump_json(),
            nx=True,
        )
    except RedisError as exc:
        raise EscalationError("Could not store escalation") from exc
    if created:
        return escalation
    existing = get_escalation(client, conversation_id, owner_token)
    if existing is None:
        raise EscalationError("Escalation disappeared during creation")
    return existing

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from redis import Redis
from redis.exceptions import RedisError, WatchError

from app.authorization import _principal_hash
from app.schemas import EscalationReason, EscalationStatus


ESCALATION_KEY_PREFIX = "getnet:escalation:conversation:"


class EscalationError(RuntimeError):
    """Raised when an escalation cannot be safely retrieved or stored."""


class EscalationOwnershipError(EscalationError):
    """Raised when the conversation belongs to another identity."""


class EscalationNotFoundError(RuntimeError):
    """Raised when an escalation does not exist."""


class EscalationTransitionError(RuntimeError):
    """Raised when an escalation transition is not permitted."""


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
    taken_by: str | None = None
    taken_at: datetime | None = None
    closed_at: datetime | None = None


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


def _read_record(value: str, key: str) -> Escalation:
    try:
        escalation = Escalation.model_validate_json(value)
    except (ValidationError, ValueError) as exc:
        raise EscalationError("Invalid escalation record") from exc
    if key != f"{ESCALATION_KEY_PREFIX}{escalation.conversation_id}":
        raise EscalationError("Invalid escalation conversation")
    return escalation


def list_escalations(client: Redis, status: EscalationStatus) -> list[Escalation]:
    records: list[Escalation] = []
    seen: set[str] = set()
    try:
        for key in client.scan_iter(match=f"{ESCALATION_KEY_PREFIX}*", count=100):
            if key in seen:
                continue
            seen.add(key)
            value = client.get(key)
            if value is None:
                raise EscalationError("Escalation disappeared during listing")
            escalation = _read_record(value, key)
            if escalation.status == status:
                records.append(escalation)
    except RedisError as exc:
        raise EscalationError("Could not list escalations") from exc
    return sorted(records, key=lambda item: (item.created_at, str(item.escalation_id)))


def find_escalation(client: Redis, escalation_id: UUID) -> tuple[str, Escalation]:
    try:
        for key in client.scan_iter(match=f"{ESCALATION_KEY_PREFIX}*", count=100):
            value = client.get(key)
            if value is None:
                raise EscalationError("Escalation disappeared during lookup")
            escalation = _read_record(value, key)
            if escalation.escalation_id == escalation_id:
                return key, escalation
    except RedisError as exc:
        raise EscalationError("Could not retrieve escalation") from exc
    raise EscalationNotFoundError("Escalation not found")


def transition_escalation(
    client: Redis, escalation_id: UUID, target: EscalationStatus, principal_id: str
) -> Escalation:
    key, _ = find_escalation(client, escalation_id)
    operator = _principal_hash(principal_id)
    try:
        for _ in range(5):
            with client.pipeline() as transaction:
                try:
                    transaction.watch(key)
                    value = transaction.get(key)
                    if value is None:
                        raise EscalationNotFoundError("Escalation not found")
                    escalation = _read_record(value, key)
                    if escalation.escalation_id != escalation_id:
                        raise EscalationNotFoundError("Escalation not found")
                    if escalation.status == target and escalation.taken_by == operator:
                        return escalation
                    now = datetime.now(timezone.utc)
                    if escalation.status == EscalationStatus.PENDING and target == EscalationStatus.TAKEN:
                        escalation.status = target
                        escalation.taken_by = operator
                        escalation.taken_at = now
                    elif (
                        escalation.status == EscalationStatus.TAKEN
                        and target == EscalationStatus.CLOSED
                        and escalation.taken_by == operator
                    ):
                        escalation.status = target
                        escalation.closed_at = now
                    else:
                        raise EscalationTransitionError("Invalid escalation transition")
                    transaction.multi()
                    transaction.set(key, escalation.model_dump_json())
                    transaction.execute()
                    return escalation
                except WatchError:
                    continue
    except RedisError as exc:
        raise EscalationError("Could not update escalation") from exc
    raise EscalationError("Escalation update conflicted; retry")

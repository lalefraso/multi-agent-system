from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, model_validator
from redis import Redis
from redis.exceptions import RedisError

from app.authorization import _principal_hash
from app.escalation import (
    Escalation,
    EscalationError,
    EscalationNotFoundError,
    EscalationTransitionError,
    find_escalation,
    list_escalations,
    reassign_escalation,
    transition_escalation,
)
from app.manage_authorization import _principal_id
from app.operator_authorization import (
    AuthorizedStaff,
    OperatorAuthorizationError,
    has_operator_access,
    require_manager,
    require_operator,
    require_operator_or_manager,
)
from app.schemas import EscalationReason, EscalationStatus
from getnet_ingestion.redis_store import RedisStoreError, _resolve_redis_url


router = APIRouter(prefix="/operator/escalations", tags=["operator"])


class EscalationItem(BaseModel):
    escalation_id: UUID
    status: EscalationStatus
    reason: EscalationReason
    created_at: datetime
    taken_at: datetime | None
    closed_at: datetime | None
    overdue: bool

    @classmethod
    def from_record(cls, record: Escalation) -> EscalationItem:
        return cls(
            escalation_id=record.escalation_id,
            status=record.status,
            reason=record.reason,
            created_at=record.created_at,
            taken_at=record.taken_at,
            closed_at=record.closed_at,
            overdue=(
                record.status == EscalationStatus.PENDING
                and datetime.now(timezone.utc) - record.created_at
                > timedelta(hours=1)
            ),
        )


class EscalationDetail(EscalationItem):
    conversation_id: UUID
    customer_id: str
    summary: str
    assigned_to: str | None
    conversation: None = None
    proposed_action: None = None
    sources: None = None

    @classmethod
    def from_record(cls, record: Escalation) -> EscalationDetail:
        return cls(
            **EscalationItem.from_record(record).model_dump(),
            conversation_id=record.conversation_id,
            customer_id=record.customer_id,
            summary=record.summary,
            assigned_to=record.taken_by,
        )


class EscalationPage(BaseModel):
    items: list[EscalationItem]
    next_offset: int | None


class EscalationStatusUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal[EscalationStatus.TAKEN, EscalationStatus.CLOSED]


class EscalationAssignmentUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    issuer: str
    subject: str

    @model_validator(mode="after")
    def validate_operator_identity(self) -> EscalationAssignmentUpdate:
        try:
            _principal_id(self.issuer, self.subject)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        return self

    def principal_id(self) -> str:
        return _principal_id(self.issuer, self.subject)


def escalation_client() -> Iterator[Redis]:
    try:
        client = Redis.from_url(_resolve_redis_url(), decode_responses=True)
    except (RedisStoreError, RedisError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not configure escalation storage",
        ) from exc
    try:
        yield client
    finally:
        try:
            client.close()
        except RedisError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Could not close escalation storage",
            ) from exc


@router.get("", response_model=EscalationPage)
def operator_list_escalations(
    status_filter: EscalationStatus = Query(
        default=EscalationStatus.PENDING, alias="status"
    ),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    staff: AuthorizedStaff = Depends(require_operator_or_manager),
    client: Redis = Depends(escalation_client),
) -> EscalationPage:
    try:
        records = list_escalations(client, status_filter)
    except EscalationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if staff.role == "operator" and status_filter != EscalationStatus.PENDING:
        records = [
            record
            for record in records
            if record.taken_by == _principal_hash(staff.principal_id)
        ]
    selected = records[offset : offset + limit]
    next_offset = (
        offset + len(selected) if offset + len(selected) < len(records) else None
    )
    return EscalationPage(
        items=[EscalationItem.from_record(record) for record in selected],
        next_offset=next_offset,
    )


@router.get("/{escalation_id}", response_model=EscalationDetail)
def operator_get_escalation(
    escalation_id: UUID,
    staff: AuthorizedStaff = Depends(require_operator_or_manager),
    client: Redis = Depends(escalation_client),
) -> EscalationDetail:
    try:
        _, record = find_escalation(client, escalation_id)
        if (
            staff.role == "operator"
            and record.status != EscalationStatus.PENDING
            and record.taken_by != _principal_hash(staff.principal_id)
        ):
            raise EscalationNotFoundError("Escalation not found")
        return EscalationDetail.from_record(record)
    except EscalationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except EscalationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.patch("/{escalation_id}/status", response_model=EscalationDetail)
def operator_update_escalation_status(
    escalation_id: UUID,
    update: EscalationStatusUpdate,
    operator: AuthorizedStaff = Depends(require_operator),
    client: Redis = Depends(escalation_client),
) -> EscalationDetail:
    try:
        record = transition_escalation(
            client, escalation_id, update.status, operator.principal_id
        )
        return EscalationDetail.from_record(record)
    except EscalationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except EscalationTransitionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except EscalationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.patch("/{escalation_id}/assignment", response_model=EscalationDetail)
def operator_reassign_escalation(
    escalation_id: UUID,
    update: EscalationAssignmentUpdate,
    _manager: AuthorizedStaff = Depends(require_manager),
    client: Redis = Depends(escalation_client),
) -> EscalationDetail:
    try:
        if not has_operator_access(client, update.principal_id()):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Target is not an authorized operator",
            )
        record = reassign_escalation(client, escalation_id, update.principal_id())
        return EscalationDetail.from_record(record)
    except EscalationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except EscalationTransitionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except OperatorAuthorizationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except EscalationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

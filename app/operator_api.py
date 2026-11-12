from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict
from redis import Redis
from redis.exceptions import RedisError

from app.auth import AuthenticatedPrincipal
from app.escalation import (
    Escalation,
    EscalationError,
    EscalationNotFoundError,
    EscalationTransitionError,
    find_escalation,
    list_escalations,
    transition_escalation,
)
from app.operator_authorization import require_operator
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

    @classmethod
    def from_record(cls, record: Escalation) -> EscalationItem:
        return cls(
            escalation_id=record.escalation_id,
            status=record.status,
            reason=record.reason,
            created_at=record.created_at,
            taken_at=record.taken_at,
            closed_at=record.closed_at,
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
    status_filter: EscalationStatus = Query(default=EscalationStatus.PENDING, alias="status"),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    _operator: AuthenticatedPrincipal = Depends(require_operator),
    client: Redis = Depends(escalation_client),
) -> EscalationPage:
    try:
        records = list_escalations(client, status_filter)
    except EscalationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    selected = records[offset:offset + limit]
    next_offset = offset + len(selected) if offset + len(selected) < len(records) else None
    return EscalationPage(
        items=[EscalationItem.from_record(record) for record in selected],
        next_offset=next_offset,
    )


@router.get("/{escalation_id}", response_model=EscalationDetail)
def operator_get_escalation(
    escalation_id: UUID,
    _operator: AuthenticatedPrincipal = Depends(require_operator),
    client: Redis = Depends(escalation_client),
) -> EscalationDetail:
    try:
        _, record = find_escalation(client, escalation_id)
        return EscalationDetail.from_record(record)
    except EscalationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except EscalationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.patch("/{escalation_id}/status", response_model=EscalationDetail)
def operator_update_escalation_status(
    escalation_id: UUID,
    update: EscalationStatusUpdate,
    operator: AuthenticatedPrincipal = Depends(require_operator),
    client: Redis = Depends(escalation_client),
) -> EscalationDetail:
    try:
        record = transition_escalation(client, escalation_id, update.status, operator.principal_id)
        return EscalationDetail.from_record(record)
    except EscalationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except EscalationTransitionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except EscalationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

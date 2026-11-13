from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from fastapi import Depends, HTTPException, status
from redis import Redis
from redis.exceptions import RedisError

from app.auth import AuthenticatedPrincipal, require_authenticated_user
from app.authorization import _principal_hash
from getnet_ingestion.redis_store import RedisStoreError, _resolve_redis_url
from getnet_ingestion.schema import MANAGER_SET_KEY, OPERATOR_SET_KEY


class OperatorAuthorizationError(RuntimeError):
    """Raised when operator authorization cannot be checked or updated."""


@dataclass(frozen=True)
class AuthorizedStaff:
    principal_id: str
    role: Literal["operator", "manager"]


def _client() -> Redis:
    try:
        return Redis.from_url(_resolve_redis_url(), decode_responses=True)
    except (RedisStoreError, RedisError, ValueError) as exc:
        raise OperatorAuthorizationError(
            "Could not configure operator authorization"
        ) from exc


def set_staff_access(
    principal_id: str, *, role: Literal["operator", "manager"], grant: bool
) -> None:
    client = _client()
    key = OPERATOR_SET_KEY if role == "operator" else MANAGER_SET_KEY
    try:
        if grant:
            client.sadd(key, _principal_hash(principal_id))
        else:
            client.srem(key, _principal_hash(principal_id))
    except RedisError as exc:
        raise OperatorAuthorizationError("Could not update staff authorization") from exc
    finally:
        try:
            client.close()
        except RedisError as exc:
            raise OperatorAuthorizationError(
                "Could not close staff authorization"
            ) from exc


def set_operator_access(principal_id: str, *, grant: bool) -> None:
    set_staff_access(principal_id, role="operator", grant=grant)


def has_operator_access(client: Redis, principal_id: str) -> bool:
    try:
        return bool(client.sismember(OPERATOR_SET_KEY, _principal_hash(principal_id)))
    except RedisError as exc:
        raise OperatorAuthorizationError("Could not verify operator authorization") from exc


def _require_staff(
    principal: AuthenticatedPrincipal = Depends(require_authenticated_user),
    *,
    roles: tuple[Literal["operator", "manager"], ...],
) -> AuthorizedStaff:
    try:
        client = _client()
        try:
            principal_hash = _principal_hash(principal.principal_id)
            memberships = {
                "manager": bool(client.sismember(MANAGER_SET_KEY, principal_hash)),
                "operator": bool(client.sismember(OPERATOR_SET_KEY, principal_hash)),
            }
        finally:
            client.close()
    except (OperatorAuthorizationError, RedisError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not verify staff authorization",
        ) from exc
    if all(memberships.values()):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Conflicting staff roles",
        )
    allowed_role = next((role for role in roles if memberships[role]), None)
    if allowed_role is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Staff access required",
        )
    return AuthorizedStaff(principal.principal_id, allowed_role)


def require_operator(
    principal: AuthenticatedPrincipal = Depends(require_authenticated_user),
) -> AuthorizedStaff:
    return _require_staff(principal, roles=("operator",))


def require_manager(
    principal: AuthenticatedPrincipal = Depends(require_authenticated_user),
) -> AuthorizedStaff:
    return _require_staff(principal, roles=("manager",))


def require_operator_or_manager(
    principal: AuthenticatedPrincipal = Depends(require_authenticated_user),
) -> AuthorizedStaff:
    return _require_staff(principal, roles=("manager", "operator"))

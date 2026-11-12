from __future__ import annotations

from fastapi import Depends, HTTPException, status
from redis import Redis
from redis.exceptions import RedisError

from app.auth import AuthenticatedPrincipal, require_authenticated_user
from app.authorization import _principal_hash
from getnet_ingestion.redis_store import RedisStoreError, _resolve_redis_url


OPERATOR_SET_KEY = "getnet:authz:operators"


class OperatorAuthorizationError(RuntimeError):
    """Raised when operator authorization cannot be checked or updated."""


def _client() -> Redis:
    try:
        return Redis.from_url(_resolve_redis_url(), decode_responses=True)
    except (RedisStoreError, RedisError, ValueError) as exc:
        raise OperatorAuthorizationError("Could not configure operator authorization") from exc


def set_operator_access(principal_id: str, *, grant: bool) -> None:
    client = _client()
    try:
        if grant:
            client.sadd(OPERATOR_SET_KEY, _principal_hash(principal_id))
        else:
            client.srem(OPERATOR_SET_KEY, _principal_hash(principal_id))
    except RedisError as exc:
        raise OperatorAuthorizationError("Could not update operator authorization") from exc
    finally:
        try:
            client.close()
        except RedisError as exc:
            raise OperatorAuthorizationError("Could not close operator authorization") from exc


def require_operator(
    principal: AuthenticatedPrincipal = Depends(require_authenticated_user),
) -> AuthenticatedPrincipal:
    try:
        client = _client()
        try:
            allowed = client.sismember(OPERATOR_SET_KEY, _principal_hash(principal.principal_id))
        finally:
            client.close()
    except (OperatorAuthorizationError, RedisError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not verify operator authorization",
        ) from exc
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Operator access required",
        )
    return principal

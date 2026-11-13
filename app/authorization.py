from __future__ import annotations

import hashlib

from redis import Redis
from redis.exceptions import RedisError

from getnet_ingestion.redis_store import RedisStoreError, _resolve_redis_url
from getnet_ingestion.schema import CUSTOMER_MEMBERSHIP_PREFIX, CUSTOMER_OWNER_PREFIX


_GRANT_CUSTOMER_SCRIPT = """
local owner = redis.call("GET", KEYS[2])
if owner and owner ~= ARGV[1] then
    redis.call("SREM", KEYS[1], ARGV[2])
    return 0
end
redis.call("SADD", KEYS[1], ARGV[2])
redis.call("SET", KEYS[2], ARGV[1])
return 1
"""

_REVOKE_CUSTOMER_SCRIPT = """
local owner = redis.call("GET", KEYS[2])
if owner ~= ARGV[1] then
    return 0
end
redis.call("SREM", KEYS[1], ARGV[2])
redis.call("DEL", KEYS[2])
return 1
"""


class CustomerAuthorizationError(RuntimeError):
    """Raised when customer access cannot be checked."""


class CustomerNotAuthorizedError(RuntimeError):
    """Raised when the principal has no access to the requested customer."""


class CustomerSelectionRequiredError(RuntimeError):
    """Raised when a principal must select one of several authorized customers."""


class CustomerAlreadyAssignedError(RuntimeError):
    """Raised when a customer is already assigned to another principal."""


class CustomerMembershipNotFoundError(RuntimeError):
    """Raised when the principal does not own the customer membership."""


def _principal_hash(principal_id: str) -> str:
    return hashlib.sha256(principal_id.encode("utf-8")).hexdigest()


def _membership_key(principal_id: str) -> str:
    return f"{CUSTOMER_MEMBERSHIP_PREFIX}{_principal_hash(principal_id)}:customers"


def _customer_owner_key(customer_id: str) -> str:
    customer_hash = hashlib.sha256(customer_id.encode("utf-8")).hexdigest()
    return f"{CUSTOMER_OWNER_PREFIX}{customer_hash}:owner"


def _run_membership_script(
    script: str, principal_id: str, customer_id: str
) -> int:
    try:
        redis_url = _resolve_redis_url()
    except RedisStoreError as exc:
        raise CustomerAuthorizationError(str(exc)) from exc

    try:
        client = Redis.from_url(redis_url, decode_responses=True)
    except (RedisError, ValueError) as exc:
        raise CustomerAuthorizationError(
            "Could not configure the Redis authorization connection"
        ) from exc

    try:
        try:
            return int(
                client.eval(
                    script,
                    2,
                    _membership_key(principal_id),
                    _customer_owner_key(customer_id),
                    _principal_hash(principal_id),
                    customer_id,
                )
            )
        except (RedisError, TypeError, ValueError) as exc:
            raise CustomerAuthorizationError(
                "Could not update customer authorization"
            ) from exc
    finally:
        try:
            client.close()
        except RedisError as exc:
            raise CustomerAuthorizationError(
                "Could not close the Redis authorization connection"
            ) from exc


def grant_customer_membership(
    principal_id: str, customer_id: str
) -> None:
    if not principal_id.strip() or not customer_id.strip():
        raise ValueError("Principal and customer IDs must not be blank")
    if len(customer_id) > 128:
        raise ValueError("Customer ID must not exceed 128 characters")
    if _run_membership_script(
        _GRANT_CUSTOMER_SCRIPT, principal_id, customer_id
    ) == 0:
        raise CustomerAlreadyAssignedError(
            "Customer is already assigned to another principal"
        )


def revoke_customer_membership(
    principal_id: str, customer_id: str
) -> None:
    if not principal_id.strip() or not customer_id.strip():
        raise ValueError("Principal and customer IDs must not be blank")
    if _run_membership_script(
        _REVOKE_CUSTOMER_SCRIPT, principal_id, customer_id
    ) == 0:
        raise CustomerMembershipNotFoundError(
            "Customer is not assigned to this principal"
        )


def authorize_customer_access(
    principal_id: str, requested_customer_id: str | None
) -> str:
    try:
        redis_url = _resolve_redis_url()
    except RedisStoreError as exc:
        raise CustomerAuthorizationError(str(exc)) from exc

    try:
        client = Redis.from_url(redis_url, decode_responses=True)
    except (RedisError, ValueError) as exc:
        raise CustomerAuthorizationError(
            "Could not configure the Redis authorization connection"
        ) from exc

    try:
        try:
            customer_ids = client.smembers(_membership_key(principal_id))
        except RedisError as exc:
            raise CustomerAuthorizationError(
                "Could not retrieve customer authorization"
            ) from exc

        if not all(
            isinstance(customer_id, str)
            and customer_id.strip()
            and len(customer_id) <= 128
            for customer_id in customer_ids
        ):
            raise CustomerAuthorizationError(
                "Invalid customer authorization mapping"
            )

        if requested_customer_id is not None:
            if requested_customer_id not in customer_ids:
                raise CustomerNotAuthorizedError
            selected_customer_id = requested_customer_id
        else:
            if not customer_ids:
                raise CustomerNotAuthorizedError
            if len(customer_ids) > 1:
                raise CustomerSelectionRequiredError
            selected_customer_id = next(iter(customer_ids))

        try:
            owner = client.get(_customer_owner_key(selected_customer_id))
        except RedisError as exc:
            raise CustomerAuthorizationError(
                "Could not verify customer ownership"
            ) from exc
        if owner != _principal_hash(principal_id):
            raise CustomerNotAuthorizedError
        return selected_customer_id
    finally:
        try:
            client.close()
        except RedisError as exc:
            raise CustomerAuthorizationError(
                "Could not close the Redis authorization connection"
            ) from exc

from __future__ import annotations

import json

from redis import Redis
from redis.exceptions import RedisError

from getnet_ingestion.redis_store import RedisStoreError, _resolve_redis_url


CUSTOMER_KEY_PREFIX = "getnet_users:customer:"


class CustomerDataError(RuntimeError):
    """Raised when customer data cannot be retrieved from Redis."""


def _redact_user_id(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _redact_user_id(item)
            for key, item in value.items()
            if key.casefold() != "user_id"
        }
    if isinstance(value, list):
        return [_redact_user_id(item) for item in value]
    return value


def fetch_customer_data(user_id: str) -> str:
    try:
        redis_url = _resolve_redis_url()
    except RedisStoreError as exc:
        raise CustomerDataError(str(exc)) from exc

    try:
        client = Redis.from_url(redis_url, decode_responses=True)
    except (RedisError, ValueError) as exc:
        raise CustomerDataError("Could not configure the Redis connection") from exc
    key = f"{CUSTOMER_KEY_PREFIX}{user_id}"
    try:
        record_type = client.type(key)
        if isinstance(record_type, bytes):
            record_type = record_type.decode("utf-8")

        if record_type == "none":
            record: object = {"found": False}
        elif record_type == "hash":
            record = client.hgetall(key)
        elif record_type == "string":
            value = client.get(key)
            if value is None:
                record = {"found": False}
            else:
                try:
                    record = json.loads(value)
                except json.JSONDecodeError:
                    record = value
        elif record_type in {"ReJSON-RL", "json"}:
            record = client.json().get(key)
        else:
            raise CustomerDataError(
                f"Unsupported Redis data type {record_type!r} for customer record"
            )
    except RedisError as exc:
        raise CustomerDataError("Could not retrieve customer data from Redis") from exc
    finally:
        client.close()

    return json.dumps(
        _redact_user_id(record), ensure_ascii=False, sort_keys=True
    )

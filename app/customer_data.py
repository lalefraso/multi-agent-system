from __future__ import annotations

import json
from datetime import date, datetime
from typing import Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from redis import Redis
from redis.exceptions import RedisError

from getnet_ingestion.redis_store import RedisStoreError, _resolve_redis_url
from getnet_ingestion.schema import (
    CUSTOMER_KEY_PREFIX,
    LEGACY_CUSTOMER_KEY_PREFIX,
    TERMINALS_KEY_PREFIX,
    TRANSACTIONS_KEY_PREFIX,
)


class CustomerDataError(RuntimeError):
    """Raised when customer data cannot be retrieved from Redis."""


class TransactionRecord(BaseModel):
    transaction_id: str
    created_at: datetime
    amount_minor: int = Field(ge=0)
    currency: str
    payment_method: str
    status: str
    decline_reason: str | None = None

    @field_validator("created_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        return value


class TerminalRecord(BaseModel):
    terminal_id: str
    model: str
    status: str
    connection_type: str
    last_seen_at: datetime

    @field_validator("last_seen_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.utcoffset() is None:
            raise ValueError("last_seen_at must include a timezone")
        return value


class PageInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    limit: int = Field(default=10, ge=1, le=50)
    offset: int = Field(default=0, ge=0)


class TransactionFilters(PageInput):
    status: Literal["approved", "declined", "pending"] | None = None
    start_date: date | None = Field(default=None, description="Inclusive local transaction date, YYYY-MM-DD")
    end_date: date | None = Field(default=None, description="Inclusive local transaction date, YYYY-MM-DD")
    payment_method: Literal["credit_card", "debit_card", "pix"] | None = None
    transaction_id: str | None = Field(default=None, min_length=1, max_length=128)

    @model_validator(mode="after")
    def check_dates(self):
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("start_date must be on or before end_date")
        return self


class TerminalFilters(PageInput):
    status: Literal["online", "offline"] | None = None
    terminal_id: str | None = Field(default=None, min_length=1, max_length=128)
    model: str | None = Field(default=None, min_length=1, max_length=128)


def _redact_identity_fields(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _redact_identity_fields(item)
            for key, item in value.items()
            if key.casefold() not in {"user_id", "customer_id", "merchant_id"}
        }
    if isinstance(value, list):
        return [_redact_identity_fields(item) for item in value]
    return value


def fetch_customer_data(customer_id: str) -> str:
    try:
        redis_url = _resolve_redis_url()
    except RedisStoreError as exc:
        raise CustomerDataError(str(exc)) from exc

    try:
        client = Redis.from_url(redis_url, decode_responses=True)
    except (RedisError, ValueError) as exc:
        raise CustomerDataError("Could not configure the Redis connection") from exc
    key = f"{CUSTOMER_KEY_PREFIX}{customer_id}"
    try:
        record_type = client.type(key)
        if isinstance(record_type, bytes):
            record_type = record_type.decode("utf-8")

        if record_type == "none":
            legacy_key = f"{LEGACY_CUSTOMER_KEY_PREFIX}{customer_id}"
            record_type = client.type(legacy_key)
            if isinstance(record_type, bytes):
                record_type = record_type.decode("utf-8")
            key = legacy_key

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
        _redact_identity_fields(record), ensure_ascii=False, sort_keys=True
    )


def fetch_account(customer_id: str) -> str:
    record = json.loads(fetch_customer_data(customer_id))
    if not isinstance(record, dict):
        raise CustomerDataError("Invalid customer account record")
    if record.get("found") is False:
        return json.dumps({"found": False})
    allowed = ("full_name", "email", "account_status", "support_tier")
    return json.dumps(
        {key: record[key] for key in allowed if key in record},
        ensure_ascii=False,
        sort_keys=True,
    )


Record = TypeVar("Record", TransactionRecord, TerminalRecord)


def _read_support_records(
    customer_id: str, prefix: str, record_type: type[Record]
) -> list[Record] | None:
    try:
        redis_url = _resolve_redis_url()
    except RedisStoreError as exc:
        raise CustomerDataError(str(exc)) from exc
    try:
        client = Redis.from_url(redis_url, decode_responses=True)
    except (RedisError, ValueError) as exc:
        raise CustomerDataError("Could not configure the Redis connection") from exc
    try:
        raw = client.get(f"{prefix}{customer_id}")
        if raw is None:
            return None
        decoded = json.loads(raw)
        if not isinstance(decoded, list):
            raise CustomerDataError("Invalid customer support dataset")
        return [record_type.model_validate(item) for item in decoded]
    except RedisError as exc:
        raise CustomerDataError("Could not retrieve customer support data from Redis") from exc
    except (json.JSONDecodeError, ValidationError) as exc:
        raise CustomerDataError("Invalid customer support dataset") from exc
    finally:
        client.close()


def fetch_transactions(customer_id: str, filters: TransactionFilters) -> str:
    records = _read_support_records(customer_id, TRANSACTIONS_KEY_PREFIX, TransactionRecord)
    if records is None:
        return json.dumps({"found": False, "records": []})
    matches = [
        record for record in records
        if (filters.status is None or record.status == filters.status)
        and (filters.payment_method is None or record.payment_method == filters.payment_method)
        and (filters.transaction_id is None or record.transaction_id == filters.transaction_id)
        and (filters.start_date is None or record.created_at.date() >= filters.start_date)
        and (filters.end_date is None or record.created_at.date() <= filters.end_date)
    ]
    return _page(matches, filters)


def fetch_terminals(customer_id: str, filters: TerminalFilters) -> str:
    records = _read_support_records(customer_id, TERMINALS_KEY_PREFIX, TerminalRecord)
    if records is None:
        return json.dumps({"found": False, "records": []})
    matches = [
        record for record in records
        if (filters.status is None or record.status == filters.status)
        and (filters.terminal_id is None or record.terminal_id == filters.terminal_id)
        and (filters.model is None or record.model.casefold() == filters.model.casefold())
    ]
    return _page(matches, filters)


def _page(records: list[TransactionRecord] | list[TerminalRecord], filters: PageInput) -> str:
    end = filters.offset + filters.limit
    return json.dumps(
        {
            "found": True,
            "records": [record.model_dump(mode="json", exclude_none=True) for record in records[filters.offset:end]],
            "next_offset": end if end < len(records) else None,
        },
        ensure_ascii=False,
    )

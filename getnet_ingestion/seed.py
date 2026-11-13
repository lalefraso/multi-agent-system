from __future__ import annotations

import argparse
import json
import sys

from redis import Redis
from redis.exceptions import RedisError

from getnet_ingestion.migrate_redis_keys import has_old_keys
from getnet_ingestion.redis_store import (
    RedisStoreError,
    _resolve_redis_url,
    embeddings_for_locale,
    persist_corpora,
)
from getnet_ingestion.schema import (
    CUSTOMER_KEY_PREFIX,
    TERMINALS_KEY_PREFIX,
    TRANSACTIONS_KEY_PREFIX,
)


SYNTHETIC_CUSTOMERS = (
    {
        "user_id": "cliente1988",
        "full_name": "Mateus Pereira",
        "email": "customer-e4d56b6c@example.com",
        "account_status": "active",
        "support_tier": "standard",
        "terminal_model": "Get Clássica",
        "last_payment_status": "paid",
        "synthetic": True,
    },
    {
        "user_id": "cliente1989",
        "full_name": "Sofia Almeida",
        "email": "customer-2a7f19c3@example.com",
        "account_status": "active",
        "support_tier": "premium",
        "terminal_model": "Get Smart",
        "last_payment_status": "paid",
        "synthetic": True,
    },
    {
        "user_id": "cliente1990",
        "full_name": "Lucas Costa",
        "email": "customer-51bc803e@example.com",
        "account_status": "pending",
        "support_tier": "standard",
        "terminal_model": "Get Clássica",
        "last_payment_status": "pending",
        "synthetic": True,
    },
    {
        "user_id": "cliente1991",
        "full_name": "Ana Ribeiro",
        "email": "customer-8d0e4b62@example.com",
        "account_status": "active",
        "support_tier": "standard",
        "terminal_model": "Get Mini",
        "last_payment_status": "failed",
        "synthetic": True,
    },
    {
        "user_id": "cliente1992",
        "full_name": "Rafael Mendes",
        "email": "customer-c3971a55@example.com",
        "account_status": "inactive",
        "support_tier": "basic",
        "terminal_model": "Get Smart",
        "last_payment_status": "paid",
        "synthetic": True,
    },
)

SYNTHETIC_TRANSACTIONS = {
    "cliente1988": [
        {
            "transaction_id": "txn_synth_001",
            "created_at": "2026-11-10T14:30:00-03:00",
            "amount_minor": 12590, "currency": "BRL",
            "payment_method": "credit_card", "status": "approved",
        },
        {
            "transaction_id": "txn_synth_002",
            "created_at": "2026-11-10T16:45:00-03:00",
            "amount_minor": 4590, "currency": "BRL",
            "payment_method": "debit_card", "status": "declined",
            "decline_reason": "insufficient_funds",
        },
        {
            "transaction_id": "txn_synth_003",
            "created_at": "2026-11-11T09:00:00-03:00",
            "amount_minor": 3000, "currency": "BRL",
            "payment_method": "pix", "status": "pending",
        },
    ],
    "cliente1989": [
        {
            "transaction_id": "txn_synth_004",
            "created_at": "2026-11-11T11:00:00-03:00",
            "amount_minor": 24900, "currency": "BRL",
            "payment_method": "credit_card", "status": "approved",
        },
    ],
    "cliente1990": [],
    "cliente1991": [
        {
            "transaction_id": "txn_synth_005",
            "created_at": "2026-11-09T12:00:00-03:00",
            "amount_minor": 7800, "currency": "BRL",
            "payment_method": "debit_card", "status": "declined",
            "decline_reason": "issuer_declined",
        },
    ],
    "cliente1992": [],
}

SYNTHETIC_TERMINALS = {
    "cliente1988": [
        {
            "terminal_id": "term_synth_001", "model": "Get Clássica",
            "status": "offline", "connection_type": "wifi",
            "last_seen_at": "2026-11-11T09:15:00-03:00",
        },
        {
            "terminal_id": "term_synth_002", "model": "Get Smart",
            "status": "online", "connection_type": "wifi",
            "last_seen_at": "2026-11-11T10:00:00-03:00",
        },
    ],
    "cliente1989": [
        {
            "terminal_id": "term_synth_003", "model": "Get Smart",
            "status": "online", "connection_type": "mobile",
            "last_seen_at": "2026-11-11T10:30:00-03:00",
        },
    ],
    "cliente1990": [],
    "cliente1991": [
        {
            "terminal_id": "term_synth_004", "model": "Get Mini",
            "status": "offline", "connection_type": "bluetooth",
            "last_seen_at": "2026-11-09T18:00:00-03:00",
        },
    ],
    "cliente1992": [],
}


class RedisSeedError(RuntimeError):
    """Raised when seed data cannot be loaded into Redis."""


def seed_synthetic_customers(client: Redis) -> tuple[int, int]:
    created = 0
    skipped = 0
    try:
        for customer in SYNTHETIC_CUSTOMERS:
            customer_id = customer["user_id"]
            inserted = client.set(
                f"{CUSTOMER_KEY_PREFIX}{customer_id}",
                json.dumps(customer, ensure_ascii=False),
                nx=True,
            )
            if inserted:
                created += 1
            else:
                skipped += 1
    except RedisError as exc:
        raise RedisSeedError(
            f"Could not seed synthetic customer records: {exc}"
        ) from exc
    return created, skipped


def seed_synthetic_support_data(client: Redis) -> tuple[int, int]:
    created = 0
    skipped = 0
    try:
        for prefix, records in (
            (TRANSACTIONS_KEY_PREFIX, SYNTHETIC_TRANSACTIONS),
            (TERMINALS_KEY_PREFIX, SYNTHETIC_TERMINALS),
        ):
            for customer_id, entries in records.items():
                inserted = client.set(
                    f"{prefix}{customer_id}",
                    json.dumps(entries, ensure_ascii=False),
                    nx=True,
                )
                if inserted:
                    created += 1
                else:
                    skipped += 1
    except RedisError as exc:
        raise RedisSeedError(f"Could not seed synthetic support data: {exc}") from exc
    return created, skipped


def seed_redis(
    redis_url: str | None = None,
    *,
    recreate_index: bool = False,
) -> tuple[int, int, int, int, int]:
    try:
        connection_url = _resolve_redis_url(redis_url)
    except RedisStoreError as exc:
        raise RedisSeedError(str(exc)) from exc

    client = Redis.from_url(connection_url, decode_responses=True)
    try:
        client.ping()
        if has_old_keys(client):
            raise RedisSeedError(
                "Old Redis keys detected. Stop all writers and migrate them "
                "with python -m getnet_ingestion.migrate_redis_keys --apply "
                "before initializing."
            )
        embeddings = tuple(embeddings_for_locale(locale) for locale in ("en", "mx"))
        try:
            sections_loaded = persist_corpora(
                embeddings,
                redis_url=connection_url,
                recreate_index=recreate_index,
            )
        except RedisStoreError as exc:
            raise RedisSeedError(
                "Could not load the Getnet knowledge index. Generate both locale "
                f"embedding files first: {exc}"
            ) from exc
        customers_created, customers_skipped = seed_synthetic_customers(client)
        support_created, support_skipped = seed_synthetic_support_data(client)
    except RedisError as exc:
        raise RedisSeedError(f"Could not connect to Redis: {exc}") from exc
    finally:
        client.close()

    return (
        customers_created, customers_skipped, sections_loaded,
        support_created, support_skipped,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Load the shared Getnet knowledge index and synthetic customer "
            "records into the configured Redis instance."
        )
    )
    parser.add_argument(
        "--redis-url",
        help="Target Redis URL (defaults to GETNET_REDIS_URL or REDIS_URL)",
    )
    parser.add_argument(
        "--recreate-index",
        action="store_true",
        help="Replace the shared Getnet index and its documents before loading",
    )
    args = parser.parse_args()

    try:
        (
            customers_created, customers_skipped, sections_loaded,
            support_created, support_skipped,
        ) = seed_redis(
            args.redis_url,
            recreate_index=args.recreate_index,
        )
    except (RedisSeedError, RedisError, ValueError) as exc:
        print(f"Redis initialization failed: {exc}", file=sys.stderr)
        return 1

    print(
        "Redis initialized: "
        f"{customers_created} synthetic customers added, "
        f"{customers_skipped} existing customer records preserved, "
        f"{support_created} support datasets added, "
        f"{support_skipped} existing support datasets preserved, "
        f"{sections_loaded} Getnet knowledge sections loaded."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

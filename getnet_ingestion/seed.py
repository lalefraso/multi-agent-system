from __future__ import annotations

import argparse
import json
import sys

from redis import Redis
from redis.exceptions import RedisError

from getnet_ingestion.redis_store import (
    RedisStoreError,
    _resolve_redis_url,
    embeddings_for_locale,
    persist_corpora,
)
from getnet_ingestion.schema import CUSTOMER_KEY_PREFIX


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


def seed_redis(
    redis_url: str | None = None,
    *,
    recreate_index: bool = False,
) -> tuple[int, int, int]:
    try:
        connection_url = _resolve_redis_url(redis_url)
    except RedisStoreError as exc:
        raise RedisSeedError(str(exc)) from exc

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

    client = Redis.from_url(connection_url, decode_responses=True)
    try:
        client.ping()
        customers_created, customers_skipped = seed_synthetic_customers(client)
    except RedisError as exc:
        raise RedisSeedError(f"Could not connect to Redis: {exc}") from exc
    finally:
        client.close()

    return customers_created, customers_skipped, sections_loaded


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
        customers_created, customers_skipped, sections_loaded = seed_redis(
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
        f"{sections_loaded} Getnet knowledge sections loaded."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

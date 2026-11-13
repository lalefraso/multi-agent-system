from __future__ import annotations

import argparse
import sys

from redis import Redis
from redis.exceptions import RedisError

from getnet_ingestion.redis_store import RedisStoreError, _resolve_redis_url
from getnet_ingestion.schema import (
    CONVERSATION_OWNER_PREFIX,
    CUSTOMER_KEY_PREFIX,
    ESCALATION_KEY_PREFIX,
    HISTORY_KEY_PREFIX,
    LEGACY_CUSTOMER_KEY_PREFIX,
    TERMINALS_KEY_PREFIX,
    TRANSACTIONS_KEY_PREFIX,
)


KEY_MAPPINGS = (
    ("customers:customer:", CUSTOMER_KEY_PREFIX),
    (LEGACY_CUSTOMER_KEY_PREFIX, CUSTOMER_KEY_PREFIX),
    ("customers:transactions:", TRANSACTIONS_KEY_PREFIX),
    ("customers:terminals:", TERMINALS_KEY_PREFIX),
    ("getnet:chat:", HISTORY_KEY_PREFIX),
    ("getnet:conversation-owner:", CONVERSATION_OWNER_PREFIX),
    ("getnet:authz:", "chat:authz:"),
    ("getnet:escalation:conversation:", ESCALATION_KEY_PREFIX),
)


class RedisMigrationError(RuntimeError):
    """Raised when Redis keys cannot be migrated safely."""


def has_old_keys(client: Redis) -> bool:
    return any(
        next(client.scan_iter(match=f"{old}*", count=100), None) is not None
        for old, _ in KEY_MAPPINGS
    )


def migrate_keys(client: Redis, *, apply: bool = False) -> int:
    try:
        moves: dict[str, str] = {}
        destinations: set[str] = set()
        for old, new in KEY_MAPPINGS:
            for source in client.scan_iter(match=f"{old}*", count=100):
                if source in moves:
                    continue
                destination = new + source[len(old):]
                if destination in destinations or client.exists(destination):
                    raise RedisMigrationError(
                        "Conflicting destination keys found; no keys were moved. "
                        "Resolve conflicts before retrying."
                    )
                moves[source] = destination
                destinations.add(destination)

        if apply:
            for source, destination in moves.items():
                if not client.renamenx(source, destination):
                    raise RedisMigrationError(
                        "A destination appeared during migration; some keys may "
                        "already have moved. Stop all writers and resolve the conflict."
                    )
        return len(moves)
    except RedisError as exc:
        raise RedisMigrationError(
            "Redis migration failed; some keys may already have moved. "
            "Keep writers stopped and retry after resolving the error."
        ) from exc


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect or migrate Redis key namespaces with all writers stopped."
    )
    parser.add_argument("--redis-url", help="Target Redis URL")
    parser.add_argument(
        "--apply", action="store_true", help="Move keys; otherwise only check"
    )
    args = parser.parse_args()
    try:
        client = Redis.from_url(_resolve_redis_url(args.redis_url), decode_responses=True)
        try:
            client.ping()
            count = migrate_keys(client, apply=args.apply)
        finally:
            client.close()
    except (RedisStoreError, RedisMigrationError, RedisError, ValueError) as exc:
        print(f"Redis migration failed: {exc}", file=sys.stderr)
        return 1
    action = "Moved" if args.apply else "Ready to move"
    print(f"{action} {count} Redis keys.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

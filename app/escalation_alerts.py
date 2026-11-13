from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from uuid import uuid4

from redis import Redis
from redis.exceptions import RedisError

from app.escalation import (
    Escalation,
    EscalationError,
    _read_record,
    list_escalations,
)
from app.schemas import EscalationStatus
from getnet_ingestion.redis_store import RedisStoreError, _resolve_redis_url
from getnet_ingestion.schema import ESCALATION_KEY_PREFIX


LOGGER = logging.getLogger(__name__)
ALERT_THRESHOLD = timedelta(hours=1)
ALERT_LOCK_TTL_SECONDS = 30
SLACK_REQUEST_TIMEOUT_SECONDS = 5
ALERT_SENT_KEY_PREFIX = "chat:escalation:alert:sent:"
ALERT_LOCK_KEY_PREFIX = "chat:escalation:alert:lock:"


class EscalationAlertError(RuntimeError):
    """Raised when an overdue escalation alert cannot be delivered safely."""


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        return None


def validate_slack_webhook_url(webhook_url: str) -> None:
    parsed = urlsplit(webhook_url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "hooks.slack.com"
        or parsed.username
        or parsed.password
        or parsed.fragment
    ):
        raise EscalationAlertError(
            "HITL Slack webhook must be an HTTPS hooks.slack.com URL"
        )


def send_slack_alert(webhook_url: str, message: str) -> None:
    validate_slack_webhook_url(webhook_url)
    request = Request(
        webhook_url,
        data=json.dumps({"text": message}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with build_opener(_NoRedirectHandler).open(
            request, timeout=SLACK_REQUEST_TIMEOUT_SECONDS
        ) as response:
            if not 200 <= response.status < 300:
                raise EscalationAlertError(
                    f"Slack delivery failed with HTTP {response.status}"
                )
    except HTTPError as exc:
        raise EscalationAlertError(
            f"Slack delivery failed with HTTP {exc.code}"
        ) from None
    except (URLError, TimeoutError, OSError):
        raise EscalationAlertError("Slack delivery failed") from None


def _is_overdue(record: Escalation, now: datetime) -> bool:
    if record.created_at.tzinfo is None or record.created_at.utcoffset() is None:
        raise EscalationAlertError("Escalation creation time must include a timezone")
    return (
        record.status == EscalationStatus.PENDING
        and now - record.created_at > ALERT_THRESHOLD
    )


def _alert_message(record: Escalation, now: datetime) -> str:
    age_minutes = int((now - record.created_at).total_seconds() // 60)
    return (
        "Aviso de escalación pendiente >1 h\n"
        f"ID: {record.escalation_id}\n"
        f"Antigüedad: {age_minutes} min\n"
        f"Motivo: {record.reason.value}"
    )


def process_overdue_escalations(
    client: Redis,
    webhook_url: str,
    *,
    now: datetime | None = None,
    sender: Callable[[str, str], None] = send_slack_alert,
) -> int:
    validate_slack_webhook_url(webhook_url)
    current_time = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None or current_time.utcoffset() is None:
        raise ValueError("now must be timezone-aware")

    delivered = 0
    for record in list_escalations(client, EscalationStatus.PENDING):
        if not _is_overdue(record, current_time):
            continue

        sent_key = f"{ALERT_SENT_KEY_PREFIX}{record.escalation_id}"
        lock_key = f"{ALERT_LOCK_KEY_PREFIX}{record.escalation_id}"
        record_key = f"{ESCALATION_KEY_PREFIX}{record.conversation_id}"
        try:
            if client.get(sent_key) is not None:
                continue
            if not client.set(
                lock_key,
                str(uuid4()),
                nx=True,
                ex=ALERT_LOCK_TTL_SECONDS,
            ):
                continue
            value = client.get(record_key)
            if value is None:
                raise EscalationAlertError(
                    "Escalation disappeared before alert delivery"
                )
            latest = _read_record(value, record_key)
            if latest.escalation_id != record.escalation_id:
                raise EscalationAlertError(
                    "Escalation changed before alert delivery"
                )
            if not _is_overdue(latest, current_time):
                continue

            sender(webhook_url, _alert_message(latest, current_time))
            if not client.set(sent_key, current_time.isoformat()):
                raise EscalationAlertError(
                    "Could not record successful escalation alert"
                )
            delivered += 1
        except RedisError as exc:
            raise EscalationAlertError(
                "Could not read or record escalation alert state"
            ) from exc

    return delivered


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Send Slack alerts for overdue human escalations."
    )
    parser.parse_args(argv)
    try:
        redis_url = _resolve_redis_url()
    except RedisStoreError as exc:
        LOGGER.error("%s", exc)
        return 1

    webhook_url = os.getenv("HITL_SLACK_WEBHOOK_URL", "").strip()
    if not webhook_url:
        LOGGER.error(
            "HITL_SLACK_WEBHOOK_URL is not configured; no alerts were sent"
        )
        return 1
    try:
        validate_slack_webhook_url(webhook_url)
        client = Redis.from_url(redis_url, decode_responses=True)
    except (EscalationAlertError, RedisError, ValueError):
        LOGGER.error("Could not configure escalation alert delivery")
        return 1

    run_succeeded = True
    try:
        delivered = process_overdue_escalations(client, webhook_url)
    except (EscalationAlertError, EscalationError) as exc:
        LOGGER.error("Escalation alert run failed (%s)", type(exc).__name__)
        run_succeeded = False
        delivered = 0
    except RedisError:
        LOGGER.error("Escalation alert run failed (RedisError)")
        run_succeeded = False
        delivered = 0
    try:
        client.close()
    except RedisError:
        LOGGER.error("Could not close escalation alert storage")
        run_succeeded = False

    LOGGER.info("Escalation alert run completed (alerts_sent=%s)", delivered)
    return 0 if run_succeeded else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())

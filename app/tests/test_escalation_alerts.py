import json
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch
from uuid import uuid4

from app.escalation import EscalationDraft, create_escalation
from app.escalation_alerts import (
    ALERT_LOCK_KEY_PREFIX,
    ALERT_SENT_KEY_PREFIX,
    EscalationAlertError,
    main,
    process_overdue_escalations,
    send_slack_alert,
    validate_slack_webhook_url,
)
from app.schemas import EscalationReason, EscalationStatus
from getnet_ingestion.schema import ESCALATION_KEY_PREFIX


class AlertRedis:
    def __init__(self):
        self.data = {}

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value, nx=False, ex=None):
        if nx and key in self.data:
            return None
        self.data[key] = value
        return True

    def scan_iter(self, match, count):
        prefix = match.removesuffix("*")
        return iter(key for key in self.data if key.startswith(prefix))


class EscalationAlertTests(unittest.TestCase):
    webhook_url = "https://hooks.slack.com/services/test/webhook"

    def setUp(self):
        self.redis = AlertRedis()
        self.now = datetime(2026, 11, 13, 16, 0, tzinfo=timezone.utc)
        self.record = create_escalation(
            self.redis,
            uuid4(),
            "owner-token",
            "private-customer-id",
            EscalationReason.HUMAN_REQUESTED,
            EscalationDraft(summary="Private customer conversation summary."),
        )
        self.record.created_at = self.now - timedelta(hours=1, minutes=5)
        self.record_key = (
            f"{ESCALATION_KEY_PREFIX}{self.record.conversation_id}"
        )
        self.redis.set(self.record_key, self.record.model_dump_json())

    def test_sends_minimal_alert_once_per_pending_case(self):
        sender = Mock()
        count = process_overdue_escalations(
            self.redis, self.webhook_url, now=self.now, sender=sender
        )
        self.assertEqual(count, 1)
        sender.assert_called_once()
        message = sender.call_args.args[1]
        self.assertIn(str(self.record.escalation_id), message)
        self.assertIn("65 min", message)
        self.assertIn("human_requested", message)
        self.assertNotIn("Private customer", message)
        self.assertNotIn("private-customer-id", message)

        count = process_overdue_escalations(
            self.redis, self.webhook_url, now=self.now, sender=sender
        )
        self.assertEqual(count, 0)
        sender.assert_called_once()
        self.assertIn(
            f"{ALERT_SENT_KEY_PREFIX}{self.record.escalation_id}",
            self.redis.data,
        )

    def test_does_not_send_at_threshold_or_for_non_pending_cases(self):
        sender = Mock()
        self.record.created_at = self.now - timedelta(hours=1)
        self.redis.set(self.record_key, self.record.model_dump_json())
        self.assertEqual(
            process_overdue_escalations(
                self.redis, self.webhook_url, now=self.now, sender=sender
            ),
            0,
        )

        self.record.created_at = self.now - timedelta(hours=2)
        self.record.status = EscalationStatus.TAKEN
        self.redis.set(self.record_key, self.record.model_dump_json())
        self.assertEqual(
            process_overdue_escalations(
                self.redis, self.webhook_url, now=self.now, sender=sender
            ),
            0,
        )
        sender.assert_not_called()

    def test_retries_after_delivery_error(self):
        sender = Mock(side_effect=EscalationAlertError("unavailable"))
        with self.assertRaises(EscalationAlertError):
            process_overdue_escalations(
                self.redis, self.webhook_url, now=self.now, sender=sender
            )
        self.assertNotIn(
            f"{ALERT_SENT_KEY_PREFIX}{self.record.escalation_id}",
            self.redis.data,
        )

        self.redis.data.pop(f"{ALERT_LOCK_KEY_PREFIX}{self.record.escalation_id}")
        sender.side_effect = None
        self.assertEqual(
            process_overdue_escalations(
                self.redis, self.webhook_url, now=self.now, sender=sender
            ),
            1,
        )

    def test_concurrent_worker_skips_case_held_by_another_worker(self):
        self.redis.set(
            f"{ALERT_LOCK_KEY_PREFIX}{self.record.escalation_id}",
            "another-worker",
            ex=30,
        )
        sender = Mock()
        self.assertEqual(
            process_overdue_escalations(
                self.redis, self.webhook_url, now=self.now, sender=sender
            ),
            0,
        )
        sender.assert_not_called()

    def test_rejects_non_slack_or_non_https_destinations(self):
        for url in (
            "http://hooks.slack.com/services/test",
            "https://example.com/webhook",
            "https://user:password@hooks.slack.com/services/test",
        ):
            with self.subTest(url=url), self.assertRaises(EscalationAlertError):
                validate_slack_webhook_url(url)

    def test_slack_sender_posts_only_message_text(self):
        response = Mock()
        response.status = 200
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        opener = Mock()
        opener.open.return_value = response
        with patch(
            "app.escalation_alerts.build_opener", return_value=opener
        ):
            send_slack_alert(self.webhook_url, "minimal alert")

        request = opener.open.call_args.args[0]
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(
            json.loads(request.data),
            {"text": "minimal alert"},
        )

    def test_missing_webhook_fails_without_sending(self):
        with (
            patch("app.escalation_alerts._resolve_redis_url", return_value="redis://test"),
            patch.dict("os.environ", {"HITL_SLACK_WEBHOOK_URL": ""}),
        ):
            self.assertEqual(main([]), 1)


if __name__ == "__main__":
    unittest.main()

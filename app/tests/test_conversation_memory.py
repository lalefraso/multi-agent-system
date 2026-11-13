import json
import unittest
from uuid import UUID
from unittest.mock import call, patch

from langchain_core.messages import AIMessage, HumanMessage
from redis.exceptions import RedisError

from app.conversation_memory import (
    CONVERSATION_LOCK_TTL_SECONDS,
    CONVERSATION_LOCK_WAIT_SECONDS,
    CONVERSATION_OWNER_PREFIX,
    CONVERSATION_OWNER_SUFFIX,
    CONVERSATION_TTL_SECONDS,
    HISTORY_KEY_PREFIX,
    MAX_CONTEXT_MESSAGES,
    ConversationBusyError,
    ConversationHistory,
    ConversationMemoryError,
    ConversationOwnershipError,
    _session_id,
)


class ConversationHistoryTests(unittest.TestCase):
    @patch("app.conversation_memory.Redis.from_url")
    @patch(
        "app.conversation_memory._resolve_redis_url",
        return_value="redis://test",
    )
    def test_configures_scoped_history_with_ttl(
        self, _resolve_url, redis_from_url
    ):
        conversation_id = UUID("9015dbdd-608a-443e-8ddb-a142817ad9df")
        history = ConversationHistory("principal-1", "customer-1", conversation_id)

        redis_from_url.assert_called_once_with(
            "redis://test", decode_responses=True
        )
        self.assertEqual(
            history._key,
            f"{HISTORY_KEY_PREFIX}"
            f"{_session_id('principal-1', 'customer-1', conversation_id)}",
        )
        self.assertEqual(
            history._owner_key,
            f"{CONVERSATION_OWNER_PREFIX}{conversation_id}{CONVERSATION_OWNER_SUFFIX}",
        )
        self.assertNotEqual(
            history._key,
            f"{HISTORY_KEY_PREFIX}"
            f"{_session_id('another-principal', 'customer-1', conversation_id)}",
        )
        history.close()
        redis_from_url.return_value.close.assert_called_once_with()

    @patch("app.conversation_memory.Redis.from_url")
    @patch(
        "app.conversation_memory._resolve_redis_url",
        return_value="redis://test",
    )
    def test_limits_loaded_context_and_saves_user_and_assistant_turn(
        self, _resolve_url, redis_from_url
    ):
        client = redis_from_url.return_value
        client.lrange.return_value = [
            f'{{"type":"human","content":"message-{i}"}}'
            for i in range(25 - MAX_CONTEXT_MESSAGES, 25)
        ]
        history = ConversationHistory(
            "principal-1",
            "customer-1",
            UUID("9015dbdd-608a-443e-8ddb-a142817ad9df"),
        )

        self.assertEqual(
            history.load_context(),
            [
                HumanMessage(content=f"message-{i}")
                for i in range(25 - MAX_CONTEXT_MESSAGES, 25)
            ],
        )
        client.lrange.assert_called_once_with(
            history._key, -MAX_CONTEXT_MESSAGES, -1
        )
        history.save_turn("new question", "new answer")
        client.pipeline.assert_called_once_with(transaction=True)
        transaction = client.pipeline.return_value.__enter__.return_value
        transaction.rpush.assert_called_once_with(
            history._key,
            '{"type": "human", "content": "new question"}',
            '{"type": "ai", "content": "new answer"}',
        )
        self.assertEqual(
            transaction.expire.call_args_list,
            [
                call(history._key, CONVERSATION_TTL_SECONDS),
                call(history._owner_key, CONVERSATION_TTL_SECONDS),
            ],
        )
        transaction.execute.assert_called_once_with()
        history.close()

    @patch("app.conversation_memory.Redis.from_url")
    @patch(
        "app.conversation_memory._resolve_redis_url",
        return_value="redis://test",
    )
    def test_acquires_and_releases_lock_for_scoped_conversation(
        self, _resolve_url, redis_from_url
    ):
        conversation_id = UUID("9015dbdd-608a-443e-8ddb-a142817ad9df")
        history = ConversationHistory(
            "principal-1", "customer-1", conversation_id
        )

        history.acquire_turn_lock()
        redis_from_url.return_value.lock.assert_called_once_with(
            f"{history._key}:lock",
            timeout=CONVERSATION_LOCK_TTL_SECONDS,
            blocking_timeout=CONVERSATION_LOCK_WAIT_SECONDS,
        )
        redis_from_url.return_value.lock.return_value.acquire.assert_called_once_with(
            blocking=True
        )

        history.release_turn_lock()

        redis_from_url.return_value.lock.return_value.release.assert_called_once_with()
        history.close()

    @patch("app.conversation_memory.Redis.from_url")
    @patch(
        "app.conversation_memory._resolve_redis_url",
        return_value="redis://test",
    )
    def test_reports_busy_conversation_when_lock_wait_times_out(
        self, _resolve_url, redis_from_url
    ):
        redis_from_url.return_value.lock.return_value.acquire.return_value = False
        history = ConversationHistory(
            "principal-1",
            "customer-1",
            UUID("9015dbdd-608a-443e-8ddb-a142817ad9df"),
        )

        with self.assertRaisesRegex(ConversationBusyError, "retry shortly"):
            history.acquire_turn_lock()

        history.release_turn_lock()
        history.close()

    @patch("app.conversation_memory.Redis.from_url")
    @patch(
        "app.conversation_memory._resolve_redis_url",
        return_value="redis://test",
    )
    def test_redis_connection_errors_are_reported(
        self, _resolve_url, redis_from_url
    ):
        redis_from_url.side_effect = RedisError("connection failed")

        with self.assertRaisesRegex(
            ConversationMemoryError, "Could not configure Redis"
        ):
            ConversationHistory(
                "principal-1",
                "customer-1",
                UUID("9015dbdd-608a-443e-8ddb-a142817ad9df"),
            )

    @patch("app.conversation_memory.Redis.from_url")
    @patch(
        "app.conversation_memory._resolve_redis_url",
        return_value="redis://test",
    )
    def test_corrupt_history_is_reported(self, _resolve_url, redis_from_url):
        redis_from_url.return_value.lrange.return_value = ["not-json"]
        history = ConversationHistory(
            "principal-1",
            "customer-1",
            UUID("9015dbdd-608a-443e-8ddb-a142817ad9df"),
        )

        with self.assertRaisesRegex(
            ConversationMemoryError, "Could not decode"
        ):
            history.load_context()

        history.close()

    @patch("app.conversation_memory.Redis.from_url")
    @patch(
        "app.conversation_memory._resolve_redis_url",
        return_value="redis://test",
    )
    def test_sanitizes_legacy_history_before_sending_it_to_agent(
        self, _resolve_url, redis_from_url
    ):
        redis_from_url.return_value.lrange.return_value = [
            json.dumps(
                {
                    "type": "human",
                    "content": (
                        '{"message":"Previous question",'
                        '"user_id":"customer-123"}'
                    ),
                }
            ),
            json.dumps(
                {"type": "ai", "content": "Your id is customer-123"}
            ),
        ]
        history = ConversationHistory(
            "principal-1",
            "customer-123",
            UUID("9015dbdd-608a-443e-8ddb-a142817ad9df"),
        )

        self.assertEqual(
            history.load_context(),
            [
                HumanMessage(content="Previous question"),
                AIMessage(content="Your id is [redacted]"),
            ],
        )
        history.close()

    @patch("app.conversation_memory.Redis.from_url")
    @patch(
        "app.conversation_memory._resolve_redis_url",
        return_value="redis://test",
    )
    def test_claims_conversation_ownership_for_principal_and_customer(
        self, _resolve_url, redis_from_url
    ):
        conversation_id = UUID("9015dbdd-608a-443e-8ddb-a142817ad9df")
        history = ConversationHistory(
            "principal-1", "customer-1", conversation_id
        )

        history.claim_ownership()

        redis_from_url.return_value.set.assert_called_once_with(
            history._owner_key,
            history._owner_token,
            nx=True,
            ex=CONVERSATION_TTL_SECONDS,
        )
        history.close()

    @patch("app.conversation_memory.Redis.from_url")
    @patch(
        "app.conversation_memory._resolve_redis_url",
        return_value="redis://test",
    )
    def test_rejects_conversation_owned_by_another_principal(
        self, _resolve_url, redis_from_url
    ):
        client = redis_from_url.return_value
        client.set.return_value = None
        client.get.return_value = "another-owner"
        history = ConversationHistory(
            "principal-1",
            "customer-1",
            UUID("9015dbdd-608a-443e-8ddb-a142817ad9df"),
        )

        with self.assertRaises(ConversationOwnershipError):
            history.claim_ownership()

        history.close()

    @patch("app.conversation_memory.Redis.from_url")
    @patch(
        "app.conversation_memory._resolve_redis_url",
        return_value="redis://test",
    )
    def test_same_conversation_id_has_separate_customer_context(
        self, _resolve_url, redis_from_url
    ):
        conversation_id = UUID("9015dbdd-608a-443e-8ddb-a142817ad9df")
        first = ConversationHistory("principal-1", "customer-1", conversation_id)
        second = ConversationHistory("principal-1", "customer-2", conversation_id)

        self.assertNotEqual(first._key, second._key)

        first.close()
        second.close()


if __name__ == "__main__":
    unittest.main()

import json
import unittest
from uuid import UUID
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage
from redis.exceptions import RedisError

from app.conversation_memory import (
    CONVERSATION_LOCK_TTL_SECONDS,
    CONVERSATION_LOCK_WAIT_SECONDS,
    CONVERSATION_TTL_SECONDS,
    HISTORY_KEY_PREFIX,
    MAX_CONTEXT_MESSAGES,
    ConversationBusyError,
    ConversationHistory,
    ConversationMemoryError,
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
        history = ConversationHistory("user-1", conversation_id)

        redis_from_url.assert_called_once_with(
            "redis://test", decode_responses=True
        )
        self.assertEqual(
            history._key,
            f"{HISTORY_KEY_PREFIX}{_session_id('user-1', conversation_id)}",
        )
        self.assertNotEqual(
            history._key,
            f"{HISTORY_KEY_PREFIX}{_session_id('another-user', conversation_id)}",
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
            "user-1", UUID("9015dbdd-608a-443e-8ddb-a142817ad9df")
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
        client.rpush.assert_called_once_with(
            history._key,
            '{"type": "human", "content": "new question"}',
            '{"type": "ai", "content": "new answer"}',
        )
        client.expire.assert_called_once_with(
            history._key, CONVERSATION_TTL_SECONDS
        )
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
        history = ConversationHistory("user-1", conversation_id)

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
            "user-1", UUID("9015dbdd-608a-443e-8ddb-a142817ad9df")
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
                "user-1", UUID("9015dbdd-608a-443e-8ddb-a142817ad9df")
            )

    @patch("app.conversation_memory.Redis.from_url")
    @patch(
        "app.conversation_memory._resolve_redis_url",
        return_value="redis://test",
    )
    def test_corrupt_history_is_reported(self, _resolve_url, redis_from_url):
        redis_from_url.return_value.lrange.return_value = ["not-json"]
        history = ConversationHistory(
            "user-1", UUID("9015dbdd-608a-443e-8ddb-a142817ad9df")
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
            "customer-123", UUID("9015dbdd-608a-443e-8ddb-a142817ad9df")
        )

        self.assertEqual(
            history.load_context(),
            [
                HumanMessage(content="Previous question"),
                AIMessage(content="Your id is [redacted]"),
            ],
        )
        history.close()


if __name__ == "__main__":
    unittest.main()

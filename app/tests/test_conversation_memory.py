import unittest
from uuid import UUID
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage
from redis.exceptions import RedisError

from app.conversation_memory import (
    CONVERSATION_TTL_SECONDS,
    HISTORY_KEY_PREFIX,
    MAX_CONTEXT_MESSAGES,
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


if __name__ == "__main__":
    unittest.main()

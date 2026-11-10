import unittest
from uuid import UUID
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage
from redis.exceptions import RedisError

from app.conversation_memory import (
    CONVERSATION_TTL_SECONDS,
    HISTORY_INDEX_NAME,
    HISTORY_KEY_PREFIX,
    MAX_CONTEXT_MESSAGES,
    ConversationHistory,
    ConversationMemoryError,
    _session_id,
)


class ConversationHistoryTests(unittest.TestCase):
    @patch("app.conversation_memory.RedisChatMessageHistory")
    @patch("app.conversation_memory.Redis.from_url")
    @patch(
        "app.conversation_memory._resolve_redis_url",
        return_value="redis://test",
    )
    def test_configures_scoped_history_with_ttl(
        self, _resolve_url, redis_from_url, history_factory
    ):
        conversation_id = UUID("9015dbdd-608a-443e-8ddb-a142817ad9df")
        history = ConversationHistory("user-1", conversation_id)

        redis_from_url.assert_called_once_with("redis://test")
        history_factory.assert_called_once()
        options = history_factory.call_args.kwargs
        self.assertEqual(options["redis_client"], redis_from_url.return_value)
        self.assertEqual(options["key_prefix"], HISTORY_KEY_PREFIX)
        self.assertEqual(options["index_name"], HISTORY_INDEX_NAME)
        self.assertEqual(options["ttl"], CONVERSATION_TTL_SECONDS)
        self.assertEqual(len(options["session_id"]), 64)
        self.assertNotEqual(
            options["session_id"],
            _session_id(
                "another-user", UUID("9015dbdd-608a-443e-8ddb-a142817ad9df")
            ),
        )
        history.close()
        redis_from_url.return_value.close.assert_called_once_with()

    @patch("app.conversation_memory.RedisChatMessageHistory")
    @patch("app.conversation_memory.Redis.from_url")
    @patch(
        "app.conversation_memory._resolve_redis_url",
        return_value="redis://test",
    )
    def test_limits_loaded_context_and_saves_user_and_assistant_turn(
        self, _resolve_url, _redis_from_url, history_factory
    ):
        messages = [HumanMessage(content=f"message-{i}") for i in range(25)]
        history_factory.return_value.messages = messages
        history = ConversationHistory(
            "user-1", UUID("9015dbdd-608a-443e-8ddb-a142817ad9df")
        )

        self.assertEqual(
            history.load_context(), messages[-MAX_CONTEXT_MESSAGES:]
        )
        history.save_turn("new question", "new answer")
        history_factory.return_value.add_messages.assert_called_once_with(
            [
                HumanMessage(content="new question"),
                AIMessage(content="new answer"),
            ]
        )
        history.close()

    @patch("app.conversation_memory.RedisChatMessageHistory")
    @patch("app.conversation_memory.Redis.from_url")
    @patch(
        "app.conversation_memory._resolve_redis_url",
        return_value="redis://test",
    )
    def test_redis_errors_are_reported_and_client_is_closed(
        self, _resolve_url, redis_from_url, history_factory
    ):
        history_factory.side_effect = RedisError("connection failed")

        with self.assertRaisesRegex(
            ConversationMemoryError, "Could not configure Redis"
        ):
            ConversationHistory(
                "user-1", UUID("9015dbdd-608a-443e-8ddb-a142817ad9df")
            )

        redis_from_url.return_value.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import hashlib
from uuid import UUID

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_redis import RedisChatMessageHistory
from redis import Redis
from redis.exceptions import RedisError

from getnet_ingestion.redis_store import RedisStoreError, _resolve_redis_url


CONVERSATION_TTL_SECONDS = 30 * 24 * 60 * 60
MAX_CONTEXT_MESSAGES = 20
HISTORY_KEY_PREFIX = "getnet:chat:"
HISTORY_INDEX_NAME = "idx:getnet:agent_chat_history"


class ConversationMemoryError(RuntimeError):
    """Raised when conversation history cannot be read or stored."""


def _session_id(user_id: str, conversation_id: UUID) -> str:
    identity = f"{user_id}\0{conversation_id}".encode("utf-8")
    return hashlib.sha256(identity).hexdigest()


class ConversationHistory:
    def __init__(self, user_id: str, conversation_id: UUID) -> None:
        try:
            redis_url = _resolve_redis_url()
        except RedisStoreError as exc:
            raise ConversationMemoryError(str(exc)) from exc

        try:
            self._client = Redis.from_url(redis_url)
            self._history = RedisChatMessageHistory(
                session_id=_session_id(user_id, conversation_id),
                redis_client=self._client,
                key_prefix=HISTORY_KEY_PREFIX,
                index_name=HISTORY_INDEX_NAME,
                ttl=CONVERSATION_TTL_SECONDS,
            )
        except (RedisError, ValueError) as exc:
            client = getattr(self, "_client", None)
            if client is not None:
                try:
                    client.close()
                except RedisError as close_exc:
                    raise ConversationMemoryError(
                        "Could not close Redis conversation history"
                    ) from close_exc
            raise ConversationMemoryError(
                "Could not configure Redis conversation history"
            ) from exc

    def load_context(self) -> list[BaseMessage]:
        try:
            return self._history.messages[-MAX_CONTEXT_MESSAGES:]
        except RedisError as exc:
            raise ConversationMemoryError(
                "Could not retrieve Redis conversation history"
            ) from exc

    def save_turn(self, user_message: str, answer: str) -> None:
        try:
            self._history.add_messages(
                [
                    HumanMessage(content=user_message),
                    AIMessage(content=answer),
                ]
            )
        except RedisError as exc:
            raise ConversationMemoryError(
                "Could not store Redis conversation history"
            ) from exc

    def close(self) -> None:
        try:
            self._client.close()
        except RedisError as exc:
            raise ConversationMemoryError(
                "Could not close Redis conversation history"
            ) from exc


def build_conversation_history(
    user_id: str, conversation_id: UUID
) -> ConversationHistory:
    return ConversationHistory(user_id, conversation_id)

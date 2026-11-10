from __future__ import annotations

import hashlib
import json
from uuid import UUID

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from redis import Redis
from redis.exceptions import RedisError

from getnet_ingestion.redis_store import RedisStoreError, _resolve_redis_url


CONVERSATION_TTL_SECONDS = 30 * 24 * 60 * 60
MAX_CONTEXT_MESSAGES = 20
HISTORY_KEY_PREFIX = "getnet:chat:"


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
            self._client = Redis.from_url(
                redis_url, decode_responses=True
            )
            self._key = (
                f"{HISTORY_KEY_PREFIX}{_session_id(user_id, conversation_id)}"
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
            records = self._client.lrange(self._key, -MAX_CONTEXT_MESSAGES, -1)
        except RedisError as exc:
            raise ConversationMemoryError(
                "Could not retrieve Redis conversation history"
            ) from exc
        messages: list[BaseMessage] = []
        try:
            for record in records:
                item = json.loads(record)
                if not isinstance(item, dict) or not isinstance(
                    item.get("content"), str
                ):
                    raise ValueError("Invalid conversation message")
                if item.get("type") == "human":
                    messages.append(HumanMessage(content=item["content"]))
                elif item.get("type") == "ai":
                    messages.append(AIMessage(content=item["content"]))
                else:
                    raise ValueError("Unsupported conversation message type")
        except (json.JSONDecodeError, ValueError) as exc:
            raise ConversationMemoryError(
                "Could not decode Redis conversation history"
            ) from exc
        return messages

    def save_turn(self, user_message: str, answer: str) -> None:
        try:
            self._client.rpush(
                self._key,
                json.dumps({"type": "human", "content": user_message}),
                json.dumps({"type": "ai", "content": answer}),
            )
            self._client.expire(self._key, CONVERSATION_TTL_SECONDS)
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

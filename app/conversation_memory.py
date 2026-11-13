from __future__ import annotations

import hashlib
import json
from uuid import UUID

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from redis import Redis
from redis.exceptions import RedisError

from app.escalation import (
    Escalation, EscalationDraft, create_escalation, get_escalation,
)
from app.schemas import EscalationReason
from getnet_ingestion.redis_store import RedisStoreError, _resolve_redis_url
from getnet_ingestion.schema import (
    CONVERSATION_OWNER_PREFIX, CONVERSATION_OWNER_SUFFIX, HISTORY_KEY_PREFIX,
)


CONVERSATION_TTL_SECONDS = 30 * 24 * 60 * 60
CONVERSATION_LOCK_TTL_SECONDS = 15 * 60
CONVERSATION_LOCK_WAIT_SECONDS = 5
MAX_CONTEXT_MESSAGES = 20
MAX_CONTEXT_CHARACTERS = 16000


class ConversationMemoryError(RuntimeError):
    """Raised when conversation history cannot be read or stored."""


class ConversationBusyError(RuntimeError):
    """Raised when another request is already handling this conversation."""


class ConversationOwnershipError(RuntimeError):
    """Raised when a principal does not own the requested conversation."""


def _session_id(
    principal_id: str, customer_id: str, conversation_id: UUID
) -> str:
    identity = (
        f"{principal_id}\0{customer_id}\0{conversation_id}".encode("utf-8")
    )
    return hashlib.sha256(identity).hexdigest()


class ConversationHistory:
    def __init__(
        self, principal_id: str, customer_id: str, conversation_id: UUID
    ) -> None:
        try:
            redis_url = _resolve_redis_url()
        except RedisStoreError as exc:
            raise ConversationMemoryError(str(exc)) from exc

        try:
            self._client = Redis.from_url(
                redis_url, decode_responses=True
            )
            self._customer_id = customer_id
            self._conversation_id = conversation_id
            owner_identity = f"{principal_id}\0{customer_id}".encode("utf-8")
            self._owner_token = hashlib.sha256(owner_identity).hexdigest()
            self._owner_key = (
                f"{CONVERSATION_OWNER_PREFIX}{conversation_id}{CONVERSATION_OWNER_SUFFIX}"
            )
            self._key = (
                f"{HISTORY_KEY_PREFIX}"
                f"{_session_id(principal_id, customer_id, conversation_id)}"
            )
            self._turn_lock = None
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

    def get_escalation(self) -> Escalation | None:
        return get_escalation(self._client, self._conversation_id, self._owner_token)

    def create_escalation(
        self, reason: EscalationReason, draft: EscalationDraft
    ) -> Escalation:
        return create_escalation(
            self._client, self._conversation_id, self._owner_token,
            self._customer_id, reason, draft,
        )

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
                content = item["content"].replace(
                    self._customer_id, "[redacted]"
                )
                if item.get("type") == "human":
                    try:
                        legacy_request = json.loads(content)
                    except json.JSONDecodeError:
                        legacy_request = None
                    if (
                        isinstance(legacy_request, dict)
                        and isinstance(legacy_request.get("message"), str)
                    ):
                        content = legacy_request["message"]
                    messages.append(HumanMessage(content=content))
                elif item.get("type") == "ai":
                    messages.append(AIMessage(content=content))
                else:
                    raise ValueError("Unsupported conversation message type")
        except (json.JSONDecodeError, ValueError) as exc:
            raise ConversationMemoryError(
                "Could not decode Redis conversation history"
            ) from exc
        compacted: list[BaseMessage] = []
        remaining = MAX_CONTEXT_CHARACTERS
        for message in reversed(messages):
            if remaining <= 0:
                break
            content = message.content
            if not isinstance(content, str):
                raise ConversationMemoryError("Invalid conversation message content")
            if len(content) > remaining:
                if compacted:
                    break
                message = message.model_copy(update={"content": content[-remaining:]})
            compacted.append(message)
            remaining -= len(message.content)
        return list(reversed(compacted))

    def claim_ownership(self) -> None:
        try:
            claimed = self._client.set(
                self._owner_key,
                self._owner_token,
                nx=True,
                ex=CONVERSATION_TTL_SECONDS,
            )
            if claimed:
                return
            owner = self._client.get(self._owner_key)
            if owner != self._owner_token:
                raise ConversationOwnershipError
            self._client.expire(self._owner_key, CONVERSATION_TTL_SECONDS)
        except ConversationOwnershipError:
            raise
        except RedisError as exc:
            raise ConversationMemoryError(
                "Could not verify Redis conversation ownership"
            ) from exc

    def acquire_turn_lock(self) -> None:
        try:
            self._turn_lock = self._client.lock(
                f"{self._key}:lock",
                timeout=CONVERSATION_LOCK_TTL_SECONDS,
                blocking_timeout=CONVERSATION_LOCK_WAIT_SECONDS,
            )
            acquired = self._turn_lock.acquire(blocking=True)
        except RedisError as exc:
            self._turn_lock = None
            raise ConversationMemoryError(
                "Could not acquire Redis conversation lock"
            ) from exc
        if not acquired:
            self._turn_lock = None
            raise ConversationBusyError(
                "This conversation is processing another request; retry shortly."
            )

    def save_turn(self, user_message: str, answer: str) -> None:
        try:
            with self._client.pipeline(transaction=True) as transaction:
                transaction.rpush(
                    self._key,
                    json.dumps({"type": "human", "content": user_message}),
                    json.dumps({"type": "ai", "content": answer}),
                )
                transaction.expire(self._key, CONVERSATION_TTL_SECONDS)
                transaction.expire(
                    self._owner_key, CONVERSATION_TTL_SECONDS
                )
                transaction.execute()
        except RedisError as exc:
            raise ConversationMemoryError(
                "Could not store Redis conversation history"
            ) from exc

    def release_turn_lock(self) -> None:
        if self._turn_lock is None:
            return
        turn_lock = self._turn_lock
        self._turn_lock = None
        try:
            turn_lock.release()
        except RedisError as exc:
            raise ConversationMemoryError(
                "Could not release Redis conversation lock"
            ) from exc

    def close(self) -> None:
        try:
            self._client.close()
        except RedisError as exc:
            raise ConversationMemoryError(
                "Could not close Redis conversation history"
            ) from exc


def build_conversation_history(
    principal_id: str, customer_id: str, conversation_id: UUID
) -> ConversationHistory:
    return ConversationHistory(principal_id, customer_id, conversation_id)

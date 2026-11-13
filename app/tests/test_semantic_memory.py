import json
import struct
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch
from uuid import UUID, uuid4

from langchain_core.messages import AIMessage
from redis.exceptions import RedisError, ResponseError, WatchError

from app.agents.customer_support import conversation_memory_tool
from app.authorization import CustomerNotAuthorizedError
from app.conversation_memory import CONVERSATION_TTL_SECONDS
from app.semantic_memory import (
    BATCH_MESSAGES, MEMORY_INDEX, MemoryScope, SemanticMemoryError,
    _redact, _summarize, _vector, read_current_summary, search_memories,
    update_summary,
)
from getnet_ingestion.embedder import EMBEDDING_DIMENSIONS
from getnet_ingestion.schema import CONVERSATION_OWNER_PREFIX, CONVERSATION_OWNER_SUFFIX


class SemanticMemoryTests(unittest.TestCase):
    def setUp(self):
        self.scope = MemoryScope("private-principal", "private-customer", uuid4())
        self.client = MagicMock()
        self.client.hmget.return_value = [None] * 5
        self.client.lrange.return_value = [
            json.dumps({"type": "human", "content": "My private-customer terminal is offline"}),
            json.dumps({"type": "ai", "content": "Check the connection"}),
        ]
        self.client.ft.return_value.info.return_value = {}
        self.transaction = self.client.pipeline.return_value.__enter__.return_value
        self.transaction.hmget.return_value = [None] * 5
        self.transaction.lrange.side_effect = lambda *_: self.client.lrange.return_value
        self.transaction.get.return_value = self.scope.owner
        for path, value in (
            ("app.semantic_memory._client", self.client),
            ("app.semantic_memory._summarize", "Terminal offline; connection suggested."),
            ("app.semantic_memory._vector", b"fake-vector"),
        ):
            item = patch(path, return_value=value)
            item.start()
            self.addCleanup(item.stop)

    def test_saves_summary_and_checkpoint_atomically_after_redaction(self):
        update_summary(self.scope)
        self.client.lrange.assert_called_once_with(self.scope.history_key, 0, BATCH_MESSAGES - 1)
        self.transaction.watch.assert_called_once_with(
            self.scope.key, self.scope.history_key,
            f"{CONVERSATION_OWNER_PREFIX}{self.scope.conversation_id}{CONVERSATION_OWNER_SUFFIX}",
        )
        self.transaction.multi.assert_called_once()
        mapping = self.transaction.hset.call_args.kwargs["mapping"]
        self.assertEqual(mapping["position"], "2")
        self.assertEqual(mapping["owner"], self.scope.owner)
        self.assertEqual(mapping["scope"], self.scope.scope_token)
        self.assertEqual(mapping["conversation_id"], str(self.scope.conversation_id))
        self.assertNotIn("private-customer", repr(mapping))
        from app.semantic_memory import _summarize
        self.assertNotIn("private-customer", repr(_summarize.call_args))
        self.transaction.expire.assert_called_once_with(self.scope.key, CONVERSATION_TTL_SECONDS)
        self.transaction.execute.assert_called_once()
        self.client.close.assert_called_once()

    def test_incremental_summary_starts_at_checkpoint_and_conflicts_do_not_commit(self):
        previous = [self.scope.scope_token, self.scope.owner, str(self.scope.conversation_id), "Old summary", "8"]
        self.client.hmget.return_value = previous
        self.transaction.hmget.return_value = previous
        update_summary(self.scope)
        self.client.lrange.assert_called_once_with(self.scope.history_key, 8, 17)
        mapping = self.transaction.hset.call_args.kwargs["mapping"]
        self.assertEqual(mapping["position"], "10")

        self.transaction.reset_mock()
        self.transaction.hmget.return_value = [self.scope.scope_token, self.scope.owner, str(self.scope.conversation_id), "new", "10"]
        with self.assertRaises(SemanticMemoryError):
            update_summary(self.scope)
        self.transaction.hset.assert_not_called()
        self.transaction.execute.assert_not_called()

    def test_owner_change_aborts_summary_update(self):
        self.transaction.get.return_value = "other-owner"
        with self.assertRaises(SemanticMemoryError):
            update_summary(self.scope)
        self.transaction.hset.assert_not_called()

    def test_failure_does_not_advance_checkpoint(self):
        self.transaction.execute.side_effect = WatchError()
        with self.assertRaises(SemanticMemoryError):
            update_summary(self.scope)
        self.transaction.execute.side_effect = None
        self.client.lrange.return_value = ['{"type":"human","content":"private"}', "invalid"]
        with self.assertRaises(SemanticMemoryError):
            update_summary(self.scope)

    def test_embedding_failure_does_not_store_summary_or_checkpoint(self):
        with patch("app.semantic_memory._vector", side_effect=SemanticMemoryError("failure")):
            with self.assertRaises(SemanticMemoryError):
                update_summary(self.scope)
        self.client.pipeline.assert_not_called()
        self.client.close.assert_called_once()

    def test_current_summary_validates_scope(self):
        self.client.hmget.return_value = [
            self.scope.scope_token, self.scope.owner, str(self.scope.conversation_id), "Summary", "2"
        ]
        self.assertEqual(read_current_summary(self.scope), "Summary")
        self.client.hmget.return_value[1] = "wrong-owner"
        with self.assertRaises(SemanticMemoryError):
            read_current_summary(self.scope)

    @patch("app.semantic_memory.authorize_customer_access")
    def test_search_filters_before_knn_checks_ownership_and_stale_checkpoint(self, authorize):
        other = MemoryScope(self.scope.principal_id, self.scope.customer_id, uuid4())
        stale = MemoryScope(self.scope.principal_id, self.scope.customer_id, uuid4())
        revoked = MemoryScope(self.scope.principal_id, self.scope.customer_id, uuid4())
        self.client.ft.return_value.search.return_value = SimpleNamespace(docs=[
            SimpleNamespace(id=self.scope.key, conversation_id=str(self.scope.conversation_id)),
            SimpleNamespace(id=revoked.key, conversation_id=str(revoked.conversation_id)),
            SimpleNamespace(id=stale.key, conversation_id=str(stale.conversation_id)),
            SimpleNamespace(id=other.key, conversation_id=str(other.conversation_id)),
        ])
        self.client.get.side_effect = lambda key: (
            "other-owner" if str(revoked.conversation_id) in key else self.scope.owner
        )
        self.client.hmget.side_effect = lambda key, *_: [
            self.scope.scope_token,
            self.scope.owner,
            str(stale.conversation_id if key == stale.key else other.conversation_id),
            "Stale" if key == stale.key else "Relevant summary",
            "2",
        ]
        self.client.llen.side_effect = lambda key: 3 if key == stale.history_key else 2
        self.assertEqual(search_memories(self.scope, "previous problem"), ["Relevant summary"])
        authorize.assert_called_once_with(self.scope.principal_id, self.scope.customer_id)
        query = self.client.ft.return_value.search.call_args.args[0]
        self.assertIn(f"@scope:{{{self.scope.scope_token}}}", query.query_string())
        self.assertIn("KNN", query.query_string())
        self.assertNotIn("private-principal", query.query_string())
        self.client.ft.assert_any_call(MEMORY_INDEX)
        self.client.get.assert_any_call(
            f"{CONVERSATION_OWNER_PREFIX}{other.conversation_id}{CONVERSATION_OWNER_SUFFIX}"
        )

    @patch("app.semantic_memory.authorize_customer_access")
    def test_revoked_authorization_fails_closed_before_search(self, authorize):
        authorize.side_effect = CustomerNotAuthorizedError()
        with self.assertRaises(SemanticMemoryError):
            search_memories(self.scope, "query")
        self.client.ft.assert_not_called()

    @patch("app.semantic_memory.authorize_customer_access")
    def test_mismatched_key_is_not_returned(self, _authorize):
        self.client.ft.return_value.search.return_value = SimpleNamespace(docs=[
            SimpleNamespace(id="chat:memory:other", conversation_id=str(uuid4()))
        ])
        with self.assertRaises(SemanticMemoryError):
            search_memories(self.scope, "query")

    def test_tool_surfaces_unavailability_without_private_error(self):
        with patch("app.agents.customer_support.search_memories", side_effect=SemanticMemoryError("private secret")):
            with self.assertLogs("app.agents.customer_support", level="WARNING") as logs:
                result = json.loads(conversation_memory_tool(self.scope).invoke({"query": "private question"}))
        self.assertEqual(result, {"available": False, "summaries": []})
        self.assertNotIn("private", repr(logs.output))

    def test_common_credentials_are_removed_before_summary(self):
        text = (
            "private-customer bearer eyJabc.abc.abc password: secret123 "
            "card 4111 1111 1111 1111"
        )
        redacted = _redact(text, self.scope)
        for secret in ("private-customer", "eyJabc", "secret123", "4111"):
            self.assertNotIn(secret, redacted)

    def test_ambiguous_identity_delimiters_disable_memory(self):
        with self.assertRaises(SemanticMemoryError):
            MemoryScope("issuer\0subject\0extra", "customer", uuid4())
        with self.assertRaises(SemanticMemoryError):
            MemoryScope("issuer\0subject", "customer\0other", uuid4())


class SummaryAndEmbeddingTests(unittest.TestCase):
    def test_summary_disables_tracing_and_falls_back(self):
        primary = Mock()
        primary.invoke.side_effect = RuntimeError("private error")
        fallback = Mock()
        fallback.invoke.return_value = AIMessage(content="A summary")
        with (
            patch("app.semantic_memory.build_chat_model",
                  return_value=(primary, "first", [("second", lambda: fallback)])),
            patch("app.semantic_memory.tracing_context") as tracing,
        ):
            self.assertEqual(_summarize("", [{"type": "human", "content": "question"}]), "A summary")
        tracing.assert_called_once_with(enabled=False)
        self.assertEqual(primary.invoke.call_count, 1)
        self.assertEqual(fallback.invoke.call_count, 1)

    def test_vector_uses_document_and_query_prefix_and_validates_dimensions(self):
        model = Mock()
        model.encode.return_value = [0.25] * EMBEDDING_DIMENSIONS
        with patch("app.agents.knowledge._embedding_model", return_value=model):
            result = _vector("summary", query=False)
            self.assertEqual(len(result), EMBEDDING_DIMENSIONS * 4)
            self.assertEqual(struct.unpack("<f", result[:4])[0], 0.25)
            self.assertEqual(model.encode.call_args.args[0], "passage: summary")
            _vector("question", query=True)
            self.assertEqual(model.encode.call_args.args[0], "query: question")
            model.encode.return_value = [0.0]
            with self.assertRaises(SemanticMemoryError):
                _vector("invalid", query=True)


if __name__ == "__main__":
    unittest.main()

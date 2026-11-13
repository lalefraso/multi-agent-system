import json
import os
import struct
import unittest
from unittest.mock import patch
from uuid import uuid4

from redis import Redis

from app.semantic_memory import MemoryScope, search_memories, update_summary
from getnet_ingestion.embedder import EMBEDDING_DIMENSIONS
from getnet_ingestion.schema import CONVERSATION_OWNER_PREFIX, CONVERSATION_OWNER_SUFFIX


@unittest.skipUnless(os.getenv("SEMANTIC_MEMORY_TEST_REDIS_URL"), "requires isolated Redis Stack")
class SemanticMemoryRedisTests(unittest.TestCase):
    def setUp(self):
        self.url = os.environ["SEMANTIC_MEMORY_TEST_REDIS_URL"]
        self.client = Redis.from_url(self.url, decode_responses=True)
        self.addCleanup(self.client.close)
        self.principal = f"synthetic-issuer\0{uuid4()}"
        self.customer = f"synthetic-customer-{uuid4()}"
        self.previous = MemoryScope(self.principal, self.customer, uuid4())
        self.current = MemoryScope(self.principal, self.customer, uuid4())
        self.other = MemoryScope("different-principal", "different-customer", uuid4())
        self.keys = []
        self.addCleanup(lambda: self.client.delete(*self.keys) if self.keys else None)

    def save_synthetic_history(self, scope, text):
        owner_key = (
            f"{CONVERSATION_OWNER_PREFIX}{scope.conversation_id}{CONVERSATION_OWNER_SUFFIX}"
        )
        self.keys.extend([scope.history_key, scope.key, owner_key])
        self.client.set(owner_key, scope.owner, ex=3600)
        self.client.rpush(
            scope.history_key,
            json.dumps({"type": "human", "content": text}),
            json.dumps({"type": "ai", "content": "Synthetic answer"}),
        )
        self.client.expire(scope.history_key, 3600)

    def test_real_index_is_scoped_and_excludes_stale_or_revoked_history(self):
        self.save_synthetic_history(self.previous, "Earlier terminal issue")
        self.save_synthetic_history(self.other, "Other customer secret")
        vector = struct.pack(f"<{EMBEDDING_DIMENSIONS}f", *([1.0] + [0.0] * (EMBEDDING_DIMENSIONS - 1)))
        with (
            patch("app.semantic_memory._resolve_redis_url", return_value=self.url),
            patch("app.semantic_memory._summarize", side_effect=lambda _, items: items[0]["content"]),
            patch("app.semantic_memory._vector", return_value=vector),
            patch("app.semantic_memory.authorize_customer_access", return_value=self.customer),
        ):
            update_summary(self.previous)
            update_summary(self.other)
            self.assertEqual(search_memories(self.current, "terminal"), ["Earlier terminal issue"])
            self.assertGreater(self.client.ttl(self.previous.key), 0)
            self.client.rpush(
                self.previous.history_key,
                json.dumps({"type": "human", "content": "A later turn"}),
            )
            self.assertEqual(search_memories(self.current, "terminal"), [])
            self.client.ltrim(self.previous.history_key, 0, 1)
            self.assertEqual(search_memories(self.current, "terminal"), ["Earlier terminal issue"])
            owner_key = (
                f"{CONVERSATION_OWNER_PREFIX}{self.previous.conversation_id}{CONVERSATION_OWNER_SUFFIX}"
            )
            self.client.delete(owner_key)
            self.assertEqual(search_memories(self.current, "terminal"), [])


if __name__ == "__main__":
    unittest.main()

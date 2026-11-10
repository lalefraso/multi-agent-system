import struct
import unittest
from unittest.mock import patch

from redis.exceptions import ResponseError

from getnet_ingestion.embedder import EMBEDDING_DIMENSIONS, MODEL_NAME
from getnet_ingestion.redis_store import (
    DEFAULT_INDEX,
    DEFAULT_PREFIX,
    LEGACY_INDEXES,
    RedisStoreError,
    _resolve_redis_url,
    embeddings_for_locale,
    load_embedding_records,
    persist_records,
    persist_corpora,
)


def embedding_record(record_id="section-1"):
    return {
        "id": record_id,
        "url": "https://www.getnet.net/en/example",
        "page_title": "Example",
        "section_heading": "Example - Payments",
        "category": "solution",
        "language": "en",
        "text": "Example - Payments\nAccept payments.",
        "embedding_model": MODEL_NAME,
        "embedding_dimensions": EMBEDDING_DIMENSIONS,
        "embedding": [0.25] + [0.0] * (EMBEDDING_DIMENSIONS - 1),
    }


class FakeIndex:
    def __init__(self, exists=False, missing_error="Unknown Index name"):
        self.exists = exists
        self.missing_error = missing_error
        self.created_schema = None

    def info(self):
        if not self.exists:
            raise ResponseError(self.missing_error)
        return {}

    def create_index(self, schema, *, definition):
        self.created_schema = (schema, definition)
        self.exists = True

    def dropindex(self, *, delete_documents):
        self.dropped_documents = delete_documents
        self.exists = False


class FakePipeline:
    def __init__(self):
        self.records = []
        self.executions = 0

    def hset(self, key, *, mapping):
        self.records.append((key, mapping))
        return self

    def execute(self):
        self.executions += 1
        return []


class FakeRedis:
    def __init__(self):
        self.index = FakeIndex()
        self.indexes = {DEFAULT_INDEX: self.index}
        self.pipeline_instance = FakePipeline()

    def ft(self, index_name):
        self.index_name = index_name
        if index_name not in self.indexes:
            self.indexes[index_name] = FakeIndex()
        return self.indexes[index_name]

    def pipeline(self, *, transaction):
        self.transaction = transaction
        return self.pipeline_instance


class RedisStoreTests(unittest.TestCase):
    @patch("getnet_ingestion.redis_store.load_dotenv")
    def test_resolves_configured_env_url_without_localhost_fallback(self, _load_dotenv):
        with patch.dict("os.environ", {"GETNET_REDIS_URL": "redis://example"}, clear=True):
            self.assertEqual(_resolve_redis_url(), "redis://example")

        with patch.dict("os.environ", {"REDIS_URL": "redis://legacy"}, clear=True):
            self.assertEqual(_resolve_redis_url(), "redis://legacy")

        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RedisStoreError, "No Redis URL configured"):
                _resolve_redis_url()

    @patch("getnet_ingestion.redis_store.load_dotenv")
    def test_explicit_redis_url_has_precedence(self, _load_dotenv):
        with patch.dict("os.environ", {"GETNET_REDIS_URL": "redis://environment"}):
            self.assertEqual(
                _resolve_redis_url("redis://argument"),
                "redis://argument",
            )

    def test_creates_vector_index_and_persists_binary_float_vectors(self):
        client = FakeRedis()

        count = persist_records([embedding_record()], client)

        self.assertEqual(count, 1)
        self.assertEqual(client.index_name, DEFAULT_INDEX)
        self.assertEqual(DEFAULT_PREFIX, "getnet:section:")
        self.assertIsNotNone(client.index.created_schema)
        self.assertFalse(client.transaction)
        self.assertEqual(client.pipeline_instance.executions, 1)
        key, stored = client.pipeline_instance.records[0]
        self.assertEqual(key, "getnet:section:section-1")
        self.assertEqual(stored["id"], "section-1")
        self.assertEqual(stored["text"], "Example - Payments\nAccept payments.")
        self.assertEqual(
            struct.unpack(f"<{EMBEDDING_DIMENSIONS}f", stored["embedding"])[0],
            0.25,
        )

    def test_reuses_existing_index(self):
        client = FakeRedis()
        client.index.exists = True

        persist_records([embedding_record()], client)

        self.assertIsNone(client.index.created_schema)

    def test_creates_index_when_server_reports_search_index_not_found(self):
        client = FakeRedis()
        client.index = FakeIndex(missing_error="SEARCH_INDEX_NOT_FOUND Index not found")
        client.indexes[DEFAULT_INDEX] = client.index

        persist_records([embedding_record()], client)

        self.assertIsNotNone(client.index.created_schema)

    def test_persists_mexico_document_into_shared_namespace(self):
        client = FakeRedis()
        record = embedding_record()
        record["id"] = "section-mx"
        record["url"] = "https://www.getnet.net/mx/example"
        record["language"] = "es"

        persist_records([record], client)

        self.assertEqual(client.index_name, DEFAULT_INDEX)
        self.assertEqual(
            client.pipeline_instance.records[0][0],
            "getnet:section:section-mx",
        )
        self.assertEqual(client.pipeline_instance.records[0][1]["language"], "es")

    def test_recreates_index_and_deletes_its_old_documents(self):
        client = FakeRedis()
        client.index = FakeIndex(exists=True)
        client.indexes[DEFAULT_INDEX] = client.index
        for legacy_index in LEGACY_INDEXES:
            client.indexes[legacy_index] = FakeIndex(exists=True)

        record_mx = embedding_record("section-mx")
        record_mx["language"] = "es"
        persist_records(
            [embedding_record(), record_mx],
            client,
            recreate_index=True,
        )

        self.assertTrue(client.index.dropped_documents)
        self.assertIsNotNone(client.index.created_schema)
        self.assertTrue(
            all(client.indexes[index].dropped_documents for index in LEGACY_INDEXES)
        )
        self.assertEqual(len(client.pipeline_instance.records), 2)

    def test_locale_helpers_select_matching_embedding_files(self):
        self.assertTrue(
            str(embeddings_for_locale("mx")).endswith("getnet_mx_embeddings.jsonl")
        )

    def test_recreate_shared_index_requires_both_locale_corpora(self):
        with self.assertRaisesRegex(RedisStoreError, "all locale corpora"):
            persist_corpora(
                (embeddings_for_locale("en"),),
                index_name=DEFAULT_INDEX,
                recreate_index=True,
            )

    def test_rejects_duplicate_ids_before_writing(self):
        client = FakeRedis()

        with self.assertRaisesRegex(RedisStoreError, "Duplicate record id"):
            persist_records([embedding_record(), embedding_record()], client)

        self.assertEqual(client.pipeline_instance.records, [])

    def test_rejects_wrong_embedding_model(self):
        record = embedding_record()
        record["embedding_model"] = "another-model"

        with self.assertRaisesRegex(RedisStoreError, "must use"):
            persist_records([record], FakeRedis())

    def test_loads_valid_jsonl_and_rejects_invalid_vectors(self):
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "embeddings.jsonl"
            input_path.write_text(
                json.dumps(embedding_record()) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(len(load_embedding_records(input_path)), 1)

            record = embedding_record()
            record["embedding"] = [float("nan")] * EMBEDDING_DIMENSIONS
            input_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RedisStoreError, "finite numeric"):
                load_embedding_records(input_path)


if __name__ == "__main__":
    unittest.main()

import struct
import unittest

from redis.exceptions import ResponseError

from getnet_ingestion.embedder import EMBEDDING_DIMENSIONS, MODEL_NAME
from getnet_ingestion.redis_store import (
    RedisStoreError,
    load_embedding_records,
    persist_records,
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
    def __init__(self, exists=False):
        self.exists = exists
        self.created_schema = None

    def info(self):
        if not self.exists:
            raise ResponseError("Unknown Index name")
        return {}

    def create_index(self, schema, *, definition):
        self.created_schema = (schema, definition)
        self.exists = True


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
        self.pipeline_instance = FakePipeline()

    def ft(self, index_name):
        self.index_name = index_name
        return self.index

    def pipeline(self, *, transaction):
        self.transaction = transaction
        return self.pipeline_instance


class RedisStoreTests(unittest.TestCase):
    def test_creates_vector_index_and_persists_binary_float_vectors(self):
        client = FakeRedis()

        count = persist_records([embedding_record()], client)

        self.assertEqual(count, 1)
        self.assertEqual(client.index_name, "idx:getnet:en")
        self.assertIsNotNone(client.index.created_schema)
        self.assertFalse(client.transaction)
        self.assertEqual(client.pipeline_instance.executions, 1)
        key, stored = client.pipeline_instance.records[0]
        self.assertEqual(key, "getnet:en:section:section-1")
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

import json
import struct
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.agents.knowledge import vector_search
from retrieval_evaluation.evaluate import EvaluationError, capture_rankings, verify_index


class RetrievalEvaluationIntegrationTests(unittest.TestCase):
    @patch("app.agents.knowledge._search_product_name_records")
    @patch("app.agents.knowledge.search_vector_records")
    def test_capture_calls_current_tool_including_product_reranking(self, vector_records, product_records):
        vector_records.return_value = [{"id": "other", "section_heading": "Other", "similarity": 0.9}]
        product_records.return_value = [{"id": "classic", "section_heading": "Get Classic"}]
        holdout = {
            "version": "test",
            "corpora": {"en": "a" * 64},
            "queries": [{"id": "q", "query": "Get Classic"}],
        }
        run = capture_rankings(
            holdout, 2,
            lambda query, depth: json.loads(
                vector_search.invoke({"query": query, "result_count": depth})
            ),
        )
        self.assertEqual(run["rankings"]["q"], ["classic", "other"])
        vector_records.assert_called_once_with("Get Classic", 2)
        product_records.assert_called_once_with("Get Classic", 2)

    @patch("getnet_ingestion.redis_store._resolve_redis_url", return_value="redis://test")
    @patch("app.agents.knowledge._knowledge_settings", return_value={"vector_index": "test-index"})
    @patch("redis.Redis.from_url")
    @patch("dotenv.load_dotenv")
    def test_verify_index_checks_keys_fields_and_vectors(self, _dotenv, from_url, _settings, _url):
        record = {
            "url": "https://example.test",
            "page_title": "Title",
            "section_heading": "Heading",
            "category": "solution",
            "language": "en",
            "text": "source",
            "embedding": [1.0] + [0.0] * 383,
        }
        fields = {
            key.encode(): value.encode()
            for key, value in record.items()
            if isinstance(value, str)
        }
        fields[b"embedding"] = struct.pack("<384f", *record["embedding"])
        client = from_url.return_value
        client.ft.return_value.search.return_value = SimpleNamespace(
            docs=[SimpleNamespace(id="getnet:section:one")], total=1
        )
        pipeline = client.pipeline.return_value
        pipeline.execute.return_value = [fields]
        verify_index({"one": record})
        pipeline.hgetall.assert_called_once_with("getnet:section:one")
        client.close.assert_called_once()

        fields[b"text"] = b"changed"
        with self.assertRaisesRegex(EvaluationError, "text mismatch"):
            verify_index({"one": record})
        fields[b"text"] = b"source"
        fields[b"embedding"] = b"changed"
        with self.assertRaisesRegex(EvaluationError, "embedding mismatch"):
            verify_index({"one": record})
        client.ft.return_value.search.return_value = SimpleNamespace(
            docs=[SimpleNamespace(id="getnet:section:other")], total=1
        )
        with self.assertRaisesRegex(EvaluationError, "do not match"):
            verify_index({"one": record})
        self.assertEqual(client.close.call_count, 4)


if __name__ == "__main__":
    unittest.main()

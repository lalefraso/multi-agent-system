import json
import struct
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.agents.knowledge import (
    KNOWLEDGE_SYSTEM_PROMPT,
    KnowledgeConfigurationError,
    _normalize_url,
    knowledge_agent,
    search_external_resources,
    search_vector_records,
    vector_search,
)


class KnowledgeSearchTests(unittest.TestCase):
    def test_prompt_requires_cross_language_product_search(self):
        self.assertIn("Search across the corpus languages", KNOWLEDGE_SYSTEM_PROMPT)
        self.assertIn(
            'search both "Get Clássica" and "Get Classic"',
            KNOWLEDGE_SYSTEM_PROMPT,
        )
        self.assertIn(
            "based only on a search in one language\nor one query.",
            KNOWLEDGE_SYSTEM_PROMPT,
        )

    @patch("app.agents.knowledge._search_product_name_records")
    @patch("app.agents.knowledge.search_vector_records")
    def test_vector_search_retrieves_candidate_for_cross_language_product_name(
        self, search_records, search_product_name_records
    ):
        search_records.return_value = [
            {
                "id": "get-mini",
                "section_heading": "Get Mini",
                "similarity": 0.9,
            },
            {
                "id": "get-smart",
                "section_heading": "Get Smart",
                "similarity": 0.85,
            },
        ]
        search_product_name_records.return_value = [
            {
                "id": "get-classic",
                "section_heading": "Get Classic",
                "url": "https://www.getnet.net/en/our-solutions/in-person-payments",
            }
        ]

        result = vector_search.invoke(
            {
                "query": "¿Cuál es la diferencia entre Get Clássica y Get Smart?",
                "result_count": 5,
            }
        )

        records = json.loads(result)
        self.assertEqual(records[0]["id"], "get-classic")
        self.assertIn("get-smart", [record["id"] for record in records])
        search_records.assert_called_once_with(
            "¿Cuál es la diferencia entre Get Clássica y Get Smart?",
            5,
        )
        search_product_name_records.assert_called_once_with("Get Classic", 5)

    def test_normalizes_urls_without_relaxing_path_allowlist(self):
        self.assertEqual(
            _normalize_url("HTTPS://Getnet.net/en/product/"),
            "https://getnet.net/en/product",
        )
        with self.assertRaises(KnowledgeConfigurationError):
            _normalize_url("http://getnet.net/en/product")
        with self.assertRaises(KnowledgeConfigurationError):
            _normalize_url("https://user:getnet.net/en/product")
        with self.assertRaises(KnowledgeConfigurationError):
            _normalize_url("https://getnet.net:8443/en/product")

    @patch("app.agents.knowledge.TavilyExtract")
    @patch("app.agents.knowledge.load_dotenv")
    @patch("app.agents.knowledge._allowed_urls")
    def test_tavily_extracts_only_explicitly_allowlisted_urls(
        self, allowed_urls, _load_dotenv, tavily_extract
    ):
        allowed_urls.return_value = ["https://getnet.net/en/product"]
        tavily_extract.return_value.invoke.return_value = {
            "results": [
                {
                    "url": "https://getnet.net/en/product/",
                    "raw_content": "Allowed content",
                },
                {
                    "url": "https://getnet.net/en/another-page",
                    "raw_content": "Unlisted same-domain content",
                },
                {
                    "url": "https://outside.example/product",
                    "raw_content": "External content",
                },
            ]
        }

        with patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"}, clear=True):
            results = search_external_resources("product features")

        self.assertEqual(
            results,
            [
                {
                    "url": "https://getnet.net/en/product",
                    "content": "Allowed content",
                    "score": None,
                }
            ],
        )
        tavily_extract.assert_called_once_with(extract_depth="basic")
        tavily_extract.return_value.invoke.assert_called_once_with(
            {
                "urls": ["https://getnet.net/en/product"],
                "query": "product features",
            }
        )

    @patch("app.agents.knowledge.TavilyExtract")
    @patch("app.agents.knowledge._allowed_urls", return_value=[])
    def test_empty_allowlist_disables_external_search(
        self, _allowed_urls, tavily_extract
    ):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(search_external_resources("search"), [])

        tavily_extract.assert_not_called()

    @patch("app.agents.knowledge.load_dotenv")
    @patch("app.agents.knowledge._allowed_urls", return_value=["https://getnet.net/en"])
    def test_tavily_requires_api_key_when_allowlist_is_configured(
        self, _allowed_urls, _load_dotenv
    ):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(
                KnowledgeConfigurationError, "TAVILY_API_KEY"
            ):
                search_external_resources("search")

    @patch("app.agents.knowledge._build_knowledge_agent")
    def test_knowledge_agent_passes_query_to_nested_agent(self, build_agent):
        invoke = Mock(
            return_value={
                "messages": [
                    SimpleNamespace(content="Grounded answer with sources.")
                ]
            }
        )
        nested_agent = SimpleNamespace(
            invoke=invoke,
        )
        build_agent.return_value = nested_agent

        answer = knowledge_agent.invoke(
            {
                "query": "What payment products are available?",
                "user_id": "customer-1",
            }
        )

        self.assertEqual(answer, "Grounded answer with sources.")
        build_agent.assert_called_once_with()
        invoke.assert_called_once_with(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "What payment products are available?",
                    }
                ]
            }
        )

    @patch("app.agents.knowledge.Redis.from_url")
    @patch("app.agents.knowledge._resolve_redis_url", return_value="redis://test")
    @patch("app.agents.knowledge.load_dotenv")
    @patch("app.agents.knowledge._embedding_model")
    @patch("app.agents.knowledge._knowledge_settings")
    def test_vector_search_uses_query_prefix_and_redis_index(
        self,
        knowledge_settings,
        embedding_model,
        _load_dotenv,
        _resolve_url,
        redis_from_url,
    ):
        class FakeRedis:
            def __init__(self):
                self.index = SimpleNamespace(
                    search=lambda query, *, query_params: self.search_result(
                        query, query_params
                    )
                )
                self.documents = [
                    SimpleNamespace(
                        id=b"getnet:section:record-1",
                        url=b"https://getnet.net/en/product",
                        page_title=b"Product",
                        section_heading=b"Product > Payments",
                        category=b"solution",
                        language=b"en",
                        text=b"Accept payments.",
                        score=b"0.125",
                    )
                ]

            def search_result(self, query, query_params):
                self.query = query
                self.query_params = query_params
                return SimpleNamespace(docs=self.documents)

            def ft(self, index_name):
                self.index_name = index_name
                return self.index

            def close(self):
                self.closed = True

        class FakeModel:
            def encode(self, text, **kwargs):
                self.text = text
                self.options = kwargs
                return [0.25] + [0.0] * 383

        fake_redis = FakeRedis()
        knowledge_settings.return_value = {
            "vector_index": "idx:getnet-test",
            "vector_result_count": 4,
        }
        embedding_model.return_value = FakeModel()
        redis_from_url.return_value = fake_redis

        results = search_vector_records("How do payments work?")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "record-1")
        self.assertEqual(results[0]["text"], "Accept payments.")
        self.assertEqual(results[0]["similarity"], 0.875)
        self.assertEqual(fake_redis.index_name, "idx:getnet-test")
        self.assertEqual(embedding_model.return_value.text, "query: How do payments work?")
        self.assertTrue(
            embedding_model.return_value.options["normalize_embeddings"]
        )
        self.assertEqual(
            struct.unpack("<384f", fake_redis.query_params["vector"])[0],
            0.25,
        )
        self.assertTrue(fake_redis.closed)


if __name__ == "__main__":
    unittest.main()

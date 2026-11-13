import json
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from app.agents.knowledge import (
    EXCHANGE_RATE_SOURCES,
    KNOWLEDGE_SYSTEM_PROMPT,
    KnowledgeConfigurationError,
    _build_knowledge_agent,
    _exchange_rate_urls,
    exchange_rate_search,
)
from app.schemas import ExchangeRateCountry, ExchangeRateSearchInput


class ExchangeRateSearchTests(unittest.TestCase):
    @patch("app.agents.knowledge.TavilyExtract")
    @patch("app.agents.knowledge._knowledge_settings", return_value={})
    def test_unconfigured_countries_are_explicitly_disabled(self, _settings, extract):
        for country, (_, pair) in EXCHANGE_RATE_SOURCES.items():
            with self.subTest(country=country):
                result = json.loads(exchange_rate_search.invoke({"country": country.value}))
                self.assertEqual(
                    result,
                    {"status": "disabled", "country": country.value, "pair": pair, "sources": []},
                )
        extract.assert_not_called()

    @patch("app.agents.knowledge._knowledge_settings")
    def test_each_country_accepts_only_its_official_host(self, settings):
        for country, (host, _) in EXCHANGE_RATE_SOURCES.items():
            with self.subTest(country=country):
                url = f"https://www.{host}/rates/"
                settings.return_value = {"exchange_rate_urls": {country.value: [url, url]}}
                self.assertEqual(_exchange_rate_urls(country), [url.rstrip("/")])
                for bad_url in (
                    f"https://other.{host}/rates",
                    f"https://{host}.outside.example/rates",
                    f"http://{host}/rates",
                    f"https://{host}:443/rates",
                    "https://127.0.0.1/rates",
                ):
                    settings.return_value = {"exchange_rate_urls": {country.value: [bad_url]}}
                    with self.assertRaises(KnowledgeConfigurationError):
                        _exchange_rate_urls(country)

    @patch("app.agents.knowledge._knowledge_settings")
    def test_invalid_configuration_fails_explicitly(self, settings):
        for configured in (
            [],
            {"unknown": []},
            {"mexico": "https://banxico.org.mx/rates"},
            {"mexico": [""]},
        ):
            settings.return_value = {"exchange_rate_urls": configured}
            with self.subTest(configured=configured), self.assertRaises(KnowledgeConfigurationError):
                _exchange_rate_urls(ExchangeRateCountry.MEXICO)

    @patch("app.agents.knowledge.TavilyExtract")
    @patch("app.agents.knowledge.load_dotenv")
    @patch("app.agents.knowledge._knowledge_settings")
    def test_only_exact_configured_pages_and_fixed_query_reach_provider(
        self, settings, _load_dotenv, extract
    ):
        settings.return_value = {
            "exchange_rate_urls": {"mexico": ["https://www.banxico.org.mx/rates"]}
        }
        extract.return_value.invoke.return_value = {
            "results": [
                {"url": "https://www.banxico.org.mx/rates/", "raw_content": "USD MXN 20, 2026-11-13"},
                {"url": "https://www.banxico.org.mx/other", "raw_content": "not allowed"},
                {"url": "https://outside.example/rates", "raw_content": "not allowed"},
            ]
        }
        with patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"}, clear=True):
            result = json.loads(exchange_rate_search.invoke({"country": "mexico"}))
        extract.return_value.invoke.assert_called_once_with(
            {
                "urls": ["https://www.banxico.org.mx/rates"],
                "query": "latest published official reference exchange rate USD MXN date",
            }
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["pair"], "USD MXN")
        self.assertEqual(
            result["sources"],
            [{"url": "https://www.banxico.org.mx/rates", "content": "USD MXN 20, 2026-11-13", "score": None}],
        )

    @patch("app.agents.knowledge.TavilyExtract")
    @patch("app.agents.knowledge.load_dotenv")
    @patch("app.agents.knowledge._knowledge_settings")
    def test_unmatched_provider_results_are_not_available(self, settings, _load_dotenv, extract):
        settings.return_value = {"exchange_rate_urls": {"spain": ["https://ecb.europa.eu/rates"]}}
        extract.return_value.invoke.return_value = {
            "results": [{"url": "https://ecb.europa.eu/other", "raw_content": "EUR USD"}]
        }
        with patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"}, clear=True):
            result = json.loads(exchange_rate_search.invoke({"country": "spain"}))
        self.assertEqual(result["status"], "no_results")
        self.assertEqual(result["sources"], [])

    @patch("app.agents.knowledge.load_dotenv")
    @patch("app.agents.knowledge._knowledge_settings")
    def test_configured_source_requires_api_key(self, settings, _load_dotenv):
        settings.return_value = {"exchange_rate_urls": {"brazil": ["https://bcb.gov.br/rates"]}}
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(KnowledgeConfigurationError, "TAVILY_API_KEY"):
                exchange_rate_search.invoke({"country": "brazil"})

    @patch("app.agents.knowledge._extract_allowed_resources", return_value=[])
    @patch("app.agents.knowledge._knowledge_settings")
    def test_each_country_uses_only_its_configured_page_and_pair(self, settings, extract):
        for country, (host, pair) in EXCHANGE_RATE_SOURCES.items():
            with self.subTest(country=country):
                url = f"https://{host}/rates"
                settings.return_value = {"exchange_rate_urls": {country.value: [url]}}
                result = json.loads(exchange_rate_search.invoke({"country": country.value}))
                extract.assert_called_with(
                    [url], f"latest published official reference exchange rate {pair} date"
                )
                self.assertEqual(result["status"], "no_results")
                self.assertEqual(result["pair"], pair)

    @patch("app.agents.knowledge.TavilyExtract")
    @patch("app.agents.knowledge.load_dotenv")
    @patch("app.agents.knowledge._knowledge_settings")
    def test_provider_error_is_not_reported_as_a_success(self, settings, _load_dotenv, extract):
        settings.return_value = {"exchange_rate_urls": {"mexico": ["https://banxico.org.mx/rates"]}}
        extract.return_value.invoke.return_value = {"error": "provider unavailable"}
        with patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "provider unavailable"):
                exchange_rate_search.invoke({"country": "mexico"})

    def test_tool_input_cannot_supply_a_url_or_unsupported_country(self):
        with self.assertRaises(ValidationError):
            ExchangeRateSearchInput(country="https://outside.example")
        with self.assertRaises(ValidationError):
            ExchangeRateSearchInput(country="mexico", url="https://outside.example")

    @patch("app.agents.knowledge.create_deep_agent")
    @patch("app.agents.knowledge.build_chat_model", return_value=(object(), "model", []))
    def test_exchange_tool_is_exposed_and_reference_constraints_are_in_prompt(
        self, _model, create_agent
    ):
        _build_knowledge_agent.cache_clear()
        try:
            _build_knowledge_agent()
            names = {tool.name for tool in create_agent.call_args.kwargs["tools"]}
            self.assertIn("exchange_rate_search", names)
            self.assertIn("not a rate\napplied by Getnet", KNOWLEDGE_SYSTEM_PROMPT)
            self.assertIn("Do not\nanswer historical", KNOWLEDGE_SYSTEM_PROMPT)
            self.assertIn("publication date and source URL", KNOWLEDGE_SYSTEM_PROMPT)
        finally:
            _build_knowledge_agent.cache_clear()


if __name__ == "__main__":
    unittest.main()

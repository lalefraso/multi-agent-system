import unittest
from types import SimpleNamespace
from unittest.mock import ANY, call, patch

from app.llm import (
    LLMConfigurationError,
    ProviderFallbackMiddleware,
    _create_chat_model,
    build_chat_model,
    check_llm_configuration,
)


class ProviderFallbackTests(unittest.TestCase):
    @patch("app.llm.load_dotenv")
    @patch("app.llm._create_chat_model")
    @patch("app.llm.load_llm_settings")
    def test_builds_selected_model_and_lazy_fallbacks(
        self, load_settings, create_chat_model, _load_dotenv
    ):
        model = object()
        load_settings.return_value = {
            "model": "openai:gpt-5.5",
            "fallback_models": ["anthropic:claude-3-5-haiku-latest", "baseten"],
            "thinking_level": "medium",
        }
        create_chat_model.return_value = model

        primary, model_name, fallbacks = build_chat_model()

        self.assertIs(primary, model)
        self.assertEqual(model_name, "openai:gpt-5.5")
        self.assertEqual(len(fallbacks), 2)
        create_chat_model.assert_called_once_with("openai:gpt-5.5", "medium")
        fallbacks[0][1]()
        fallbacks[1][1]()
        self.assertEqual(
            create_chat_model.call_args_list[1:],
            [
                call(
                    "anthropic:claude-3-5-haiku-latest",
                    None,
                ),
                call("baseten", None),
            ],
        )

    def test_calls_models_in_order_until_one_succeeds(self):
        class FakeRequest(SimpleNamespace):
            def override(self, **kwargs):
                return FakeRequest(**(vars(self) | kwargs))

        openai, claude, baseten = object(), object(), object()
        request = FakeRequest(
            model=openai,
            messages=["query"],
            tools=[
                SimpleNamespace(name="knowledge_agent"),
                SimpleNamespace(name="customer_support_agent"),
                SimpleNamespace(name="write_file"),
            ],
        )
        called = []

        def handler(model_request):
            called.append(model_request.model)
            self.assertEqual(
                [tool.name for tool in model_request.tools],
                ["knowledge_agent", "customer_support_agent"],
            )
            if model_request.model is not baseten:
                raise RuntimeError("provider unavailable")
            return "response"

        middleware = ProviderFallbackMiddleware(
            "openai:gpt-5.4",
            [
                ("anthropic:claude", lambda: claude),
                ("baseten", lambda: baseten),
            ],
            {"knowledge_agent", "customer_support_agent"},
        )
        result = middleware.wrap_model_call(request, handler)

        self.assertEqual(result, "response")
        self.assertEqual(called, [openai, claude, baseten])

    @patch("app.llm.LOGGER")
    def test_logs_attempts_without_logging_prompt_or_response(self, logger):
        class FakeRequest(SimpleNamespace):
            def override(self, **kwargs):
                return FakeRequest(**(vars(self) | kwargs))

        request = FakeRequest(
            model=object(),
            messages=["private customer prompt"],
            tools=[],
        )
        response = SimpleNamespace(content="private customer response")
        middleware = ProviderFallbackMiddleware("openai:gpt-5.4", [], set())

        result = middleware.wrap_model_call(request, lambda _request: response)

        self.assertIs(result, response)
        logger.info.assert_any_call(
            "Calling model provider %s (attempt %d, messages=%d)",
            "openai:gpt-5.4",
            1,
            1,
        )
        logger.info.assert_any_call(
            "Model provider %s succeeded (attempt %d, duration_ms=%.0f)",
            "openai:gpt-5.4",
            1,
            ANY,
        )
        logged_values = repr(logger.mock_calls)
        self.assertNotIn("private customer prompt", logged_values)
        self.assertNotIn("private customer response", logged_values)

    @patch("app.llm.init_chat_model")
    def test_uses_init_chat_model_with_selected_provider_and_model(self, init_model):
        with patch.dict("os.environ", {}, clear=True):
            _create_chat_model("openai:gpt-5.4")

        init_model.assert_called_once_with(
            model="openai:gpt-5.4",
            timeout=30,
            max_retries=1,
        )

    @patch("app.llm.init_chat_model")
    def test_resolves_baseten_fallback_from_environment(self, init_model):
        with patch.dict(
            "os.environ",
            {
                "BASETEN_API_KEY": "test-key",
                "BASETEN_BASE_URL": "https://example.baseten.co/v1",
                "BASETEN_MODEL": "deployment-id",
            },
            clear=True,
        ):
            _create_chat_model("baseten")

        init_model.assert_called_once_with(
            model="openai:deployment-id",
            base_url="https://example.baseten.co/v1",
            api_key="test-key",
            timeout=30,
            max_retries=1,
        )

    @patch("app.llm.init_chat_model")
    def test_resolves_provider_model_baseten_fallback(self, init_model):
        with patch.dict(
            "os.environ",
            {
                "BASETEN_API_KEY": "test-key",
                "BASETEN_BASE_URL": "https://example.baseten.co/v1",
            },
            clear=True,
        ):
            _create_chat_model("baseten:moonshotai/Kimi-K2.6")

        init_model.assert_called_once_with(
            model="openai:moonshotai/Kimi-K2.6",
            base_url="https://example.baseten.co/v1",
            api_key="test-key",
            timeout=30,
            max_retries=1,
        )

    @patch("app.llm._create_chat_model")
    def test_configured_fallbacks_use_supported_model_identifiers(
        self, create_chat_model
    ):
        create_chat_model.return_value = object()

        primary, model_name, fallbacks = build_chat_model()

        self.assertIsNotNone(primary)
        self.assertEqual(model_name, "baseten:deepseek-ai/DeepSeek-V4.1-Flash")
        self.assertEqual(
            [model_name for model_name, _ in fallbacks],
            [
                "openai:gpt-6-luna",
                "anthropic:claude-sonnet-4-6",
            ],
        )
        for _, create_fallback in fallbacks:
            create_fallback()

        self.assertEqual(
            [call_args.args[0] for call_args in create_chat_model.call_args_list],
            [
                "baseten:deepseek-ai/DeepSeek-V4.1-Flash",
                "openai:gpt-6-luna",
                "anthropic:claude-sonnet-4-6",
            ],
        )

    @patch("app.llm.init_chat_model")
    def test_passes_thinking_level_for_google_genai(self, init_model):
        _create_chat_model("google_genai:gemini-3.6-flash", "medium")

        init_model.assert_called_once_with(
            model="google_genai:gemini-3.6-flash",
            timeout=30,
            max_retries=1,
            thinking_level="medium",
        )

    @patch("app.llm._create_chat_model")
    @patch("app.llm.load_llm_settings")
    @patch("app.llm._load_environment")
    def test_checks_primary_provider_configuration_without_network_request(
        self, _load_environment, load_settings, create_chat_model
    ):
        load_settings.return_value = {
            "model": "openai:gpt-5.5",
            "fallback_models": [],
            "thinking_level": None,
        }

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=True):
            check_llm_configuration()

        create_chat_model.assert_called_once_with("openai:gpt-5.5", None)

    @patch("app.llm.load_llm_settings")
    @patch("app.llm._load_environment")
    def test_rejects_missing_primary_provider_credentials(
        self, _load_environment, load_settings
    ):
        load_settings.return_value = {
            "model": "anthropic:claude",
            "fallback_models": [],
        }

        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(
                LLMConfigurationError, "ANTHROPIC_API_KEY"
            ):
                check_llm_configuration()


if __name__ == "__main__":
    unittest.main()

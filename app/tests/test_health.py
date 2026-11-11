import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from redis.exceptions import RedisError, ResponseError

from app.llm import LLMConfigurationError
from app.main import _check_redis_dependencies, app


class HealthTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch("app.main.check_llm_configuration")
    @patch(
        "app.main._check_redis_dependencies",
        return_value={"redis": "ok", "rag_index": "ok"},
    )
    def test_health_is_ok_when_dependencies_are_ready(
        self, _check_redis, _check_llm
    ):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "status": "ok",
                "checks": {
                    "redis": "ok",
                    "rag_index": "ok",
                    "llm_provider": "ok",
                },
            },
        )

    @patch("app.main.check_llm_configuration")
    @patch(
        "app.main._check_redis_dependencies",
        return_value={"redis": "ok", "rag_index": "unavailable"},
    )
    def test_health_returns_service_unavailable_when_a_check_fails(
        self, _check_redis, check_llm
    ):
        check_llm.side_effect = LLMConfigurationError(
            "Missing provider configuration"
        )

        response = self.client.get("/health")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["status"], "unavailable")
        self.assertEqual(
            response.json()["checks"],
            {
                "redis": "ok",
                "rag_index": "unavailable",
                "llm_provider": "unavailable",
            },
        )

    @patch("app.main._resolve_redis_url", return_value="redis://localhost:6379")
    @patch(
        "app.main._knowledge_settings",
        return_value={"vector_index": "idx:getnet"},
    )
    @patch("app.main.Redis.from_url")
    def test_redis_health_checks_ping_and_configured_vector_index(
        self, from_url, _knowledge_settings, _resolve_redis_url
    ):
        redis_client = from_url.return_value

        checks = _check_redis_dependencies()

        self.assertEqual(checks, {"redis": "ok", "rag_index": "ok"})
        redis_client.ping.assert_called_once_with()
        redis_client.ft.assert_called_once_with("idx:getnet")
        redis_client.ft.return_value.info.assert_called_once_with()
        redis_client.close.assert_called_once_with()
        from_url.assert_called_once_with(
            "redis://localhost:6379",
            decode_responses=True,
            socket_connect_timeout=1,
            socket_timeout=1,
        )

    @patch("app.main._resolve_redis_url", return_value="redis://localhost:6379")
    @patch(
        "app.main._knowledge_settings",
        return_value={"vector_index": "idx:getnet"},
    )
    @patch("app.main.Redis.from_url")
    def test_missing_vector_index_marks_only_rag_unavailable(
        self, from_url, _knowledge_settings, _resolve_redis_url
    ):
        from_url.return_value.ft.return_value.info.side_effect = ResponseError(
            "Unknown Index name"
        )

        checks = _check_redis_dependencies()

        self.assertEqual(checks, {"redis": "ok", "rag_index": "unavailable"})

    @patch("app.main._resolve_redis_url", return_value="redis://localhost:6379")
    @patch("app.main.Redis.from_url")
    def test_redis_connection_failure_marks_redis_and_index_unavailable(
        self, from_url, _resolve_redis_url
    ):
        from_url.return_value.ping.side_effect = RedisError("offline")

        checks = _check_redis_dependencies()

        self.assertEqual(
            checks, {"redis": "unavailable", "rag_index": "unavailable"}
        )


if __name__ == "__main__":
    unittest.main()

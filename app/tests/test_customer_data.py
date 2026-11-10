import json
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, ToolMessage
from redis.exceptions import RedisError

from app.main import chat
from app.customer_data import CustomerDataError, fetch_customer_data
from app.schemas import AgentName, ChatRequest


class CustomerDataTests(unittest.TestCase):
    @patch("app.customer_data.Redis.from_url")
    @patch("app.customer_data._resolve_redis_url", return_value="redis://test")
    def test_reads_all_fields_from_the_user_hash(self, _resolve_url, redis_from_url):
        client = redis_from_url.return_value
        client.type.return_value = "hash"
        client.hgetall.return_value = {
            "user_id": "cliente1988",
            "account_status": "active",
        }

        result = fetch_customer_data("cliente1988")

        self.assertEqual(
            json.loads(result),
            {"user_id": "cliente1988", "account_status": "active"},
        )
        redis_from_url.assert_called_once_with(
            "redis://test", decode_responses=True
        )
        client.type.assert_called_once_with(
            "getnet_users:customer:cliente1988"
        )
        client.hgetall.assert_called_once_with(
            "getnet_users:customer:cliente1988"
        )
        client.close.assert_called_once_with()

    @patch("app.customer_data.Redis.from_url")
    @patch("app.customer_data._resolve_redis_url", return_value="redis://test")
    def test_returns_decoded_json_string_records(self, _resolve_url, redis_from_url):
        client = redis_from_url.return_value
        client.type.return_value = "string"
        client.get.return_value = '{"account_status":"active"}'

        result = fetch_customer_data("cliente1988")

        self.assertEqual(json.loads(result), {"account_status": "active"})
        client.close.assert_called_once_with()

    @patch("app.customer_data.Redis.from_url")
    @patch("app.customer_data._resolve_redis_url", return_value="redis://test")
    def test_returns_redis_json_records(self, _resolve_url, redis_from_url):
        client = redis_from_url.return_value
        client.type.return_value = "ReJSON-RL"
        client.json.return_value.get.return_value = {"account_status": "active"}

        result = fetch_customer_data("cliente1988")

        self.assertEqual(json.loads(result), {"account_status": "active"})
        client.json.return_value.get.assert_called_once_with(
            "getnet_users:customer:cliente1988"
        )
        client.close.assert_called_once_with()

    @patch("app.customer_data.Redis.from_url")
    @patch("app.customer_data._resolve_redis_url", return_value="redis://test")
    def test_returns_explicit_result_when_user_record_is_missing(
        self, _resolve_url, redis_from_url
    ):
        client = redis_from_url.return_value
        client.type.return_value = "none"

        result = fetch_customer_data("unknown-user")

        self.assertEqual(
            json.loads(result),
            {"found": False, "user_id": "unknown-user"},
        )
        client.close.assert_called_once_with()

    @patch("app.customer_data.Redis.from_url")
    @patch("app.customer_data._resolve_redis_url", return_value="redis://test")
    def test_reports_redis_errors_and_closes_connection(
        self, _resolve_url, redis_from_url
    ):
        client = redis_from_url.return_value
        client.type.side_effect = RedisError("connection failed")

        with self.assertRaisesRegex(CustomerDataError, "Could not retrieve"):
            fetch_customer_data("cliente1988")

        client.close.assert_called_once_with()

    @patch("app.customer_data.Redis.from_url")
    @patch("app.customer_data._resolve_redis_url", return_value="redis://test")
    def test_rejects_unsupported_record_types_and_closes_connection(
        self, _resolve_url, redis_from_url
    ):
        client = redis_from_url.return_value
        client.type.return_value = "list"

        with self.assertRaisesRegex(CustomerDataError, "Unsupported Redis data type"):
            fetch_customer_data("cliente1988")

        client.close.assert_called_once_with()

    @patch("app.main._router_agent")
    def test_customer_only_response_returns_the_query_output(self, router_agent):
        lookup_output = '{"account_status":"active","user_id":"cliente1988"}'
        router_agent.return_value.invoke.return_value = {
            "messages": [
                ToolMessage(
                    content=lookup_output,
                    tool_call_id="customer-lookup",
                    name="customer_support_agent",
                ),
                AIMessage(content="The account is active."),
            ]
        }

        response = chat(
            ChatRequest(
                message="Show my account details",
                user_id="cliente1988",
            )
        )

        self.assertEqual(response.answer, lookup_output)
        self.assertEqual(response.agents_used, [AgentName.CUSTOMER_SUPPORT])


if __name__ == "__main__":
    unittest.main()

import json
import unittest
from unittest.mock import Mock, patch
from uuid import UUID

from fastapi import HTTPException
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agents.customer_support import customer_support_tool
from app.auth import AuthenticatedPrincipal
from app.authorization import (
    CustomerNotAuthorizedError,
    CustomerSelectionRequiredError,
)
from redis.exceptions import RedisError

from app.main import chat
from app.conversation_memory import (
    ConversationBusyError,
    ConversationMemoryError,
    ConversationOwnershipError,
)
from app.customer_data import CustomerDataError, fetch_customer_data
from app.schemas import AgentName, ChatRequest


class CustomerDataTests(unittest.TestCase):
    def setUp(self):
        authorization = patch(
            "app.main.authorize_customer_access",
            return_value="customer-1",
        )
        self.authorize_customer = authorization.start()
        self.addCleanup(authorization.stop)

    @patch("app.agents.customer_support.build_customer_support_agent")
    def test_customer_agent_is_bound_to_authorized_customer(self, build_agent):
        build_agent.return_value.invoke.return_value = {
            "messages": [AIMessage(content="Account is active.")]
        }
        tool = customer_support_tool("verified-customer-1")

        self.assertEqual(set(tool.args), {"query"})
        self.assertEqual(tool.invoke({"query": "Show my account"}), "Account is active.")
        build_agent.assert_called_once_with("verified-customer-1")

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
            {"account_status": "active"},
        )
        redis_from_url.assert_called_once_with(
            "redis://test", decode_responses=True
        )
        client.type.assert_called_once_with(
            "customers:customer:cliente1988"
        )
        client.hgetall.assert_called_once_with(
            "customers:customer:cliente1988"
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
            "customers:customer:cliente1988"
        )
        client.close.assert_called_once_with()

    @patch("app.customer_data.Redis.from_url")
    @patch("app.customer_data._resolve_redis_url", return_value="redis://test")
    def test_reads_legacy_customer_key_when_new_key_is_missing(
        self, _resolve_url, redis_from_url
    ):
        client = redis_from_url.return_value
        client.type.side_effect = ["none", "string"]
        client.get.return_value = '{"account_status":"active"}'

        result = fetch_customer_data("cliente1988")

        self.assertEqual(json.loads(result), {"account_status": "active"})
        self.assertEqual(
            [call.args[0] for call in client.type.call_args_list],
            [
                "customers:customer:cliente1988",
                "getnet_users:customer:cliente1988",
            ],
        )
        client.close.assert_called_once_with()

    @patch("app.customer_data.Redis.from_url")
    @patch("app.customer_data._resolve_redis_url", return_value="redis://test")
    def test_redacts_user_id_fields_from_customer_data(
        self, _resolve_url, redis_from_url
    ):
        client = redis_from_url.return_value
        client.type.return_value = "string"
        client.get.return_value = (
            '{"user_id":"customer-123","profile":{"CUSTOMER_ID":"customer-123",'
            '"merchant_id":"merchant-123","account_status":"active"}}'
        )

        result = fetch_customer_data("customer-123")

        self.assertEqual(
            json.loads(result),
            {"profile": {"account_status": "active"}},
        )

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
            {"found": False},
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

    @patch("app.main.build_conversation_history")
    @patch("app.main._router_agent")
    def test_customer_only_response_returns_the_query_output(
        self, router_agent, build_history
    ):
        history = Mock()
        history.load_context.return_value = []
        build_history.return_value = history
        lookup_output = '{"account_status":"active"}'
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
                customer_id="customer-1",
            ),
            AuthenticatedPrincipal(principal_id="principal-1"),
        )

        self.assertEqual(response.answer, lookup_output)
        self.assertEqual(response.agents_used, [AgentName.CUSTOMER_SUPPORT])
        self.assertIsNotNone(response.conversation_id)
        self.authorize_customer.assert_called_once_with(
            "principal-1", "customer-1"
        )
        router_agent.assert_called_once_with("customer-1")
        history.acquire_turn_lock.assert_called_once_with()
        build_history.assert_called_once_with(
            "principal-1", "customer-1", response.conversation_id
        )
        history.save_turn.assert_called_once_with(
            "Show my account details",
            lookup_output,
        )
        history.release_turn_lock.assert_called_once_with()
        history.close.assert_called_once_with()

    @patch("app.main.build_conversation_history")
    @patch("app.main._router_agent")
    def test_chat_uses_prior_turns_and_returns_supplied_conversation_id(
        self, router_agent, build_history
    ):
        conversation_id = UUID("9015dbdd-608a-443e-8ddb-a142817ad9df")
        previous_messages = [
            HumanMessage(content='{"message":"Earlier question","user_id":"user-1"}'),
            AIMessage(content="Earlier answer"),
        ]
        history = Mock()
        history.load_context.return_value = previous_messages.copy()
        build_history.return_value = history
        router_agent.return_value.invoke.return_value = {
            "messages": [*previous_messages, AIMessage(content="Follow-up answer")]
        }

        response = chat(
            ChatRequest(
                message="Follow up",
                conversation_id=conversation_id,
            ),
            AuthenticatedPrincipal(principal_id="principal-1"),
        )

        self.assertEqual(response.conversation_id, conversation_id)
        self.assertEqual(response.answer, "Follow-up answer")
        router_agent.return_value.invoke.assert_called_once_with(
            {
                "messages": [
                    *previous_messages,
                    HumanMessage(content="Follow up"),
                ]
            }
        )
        history.save_turn.assert_called_once_with(
            "Follow up",
            "Follow-up answer",
        )
        history.close.assert_called_once_with()

    @patch(
        "app.main.build_conversation_history",
        side_effect=ConversationMemoryError("Redis unavailable"),
    )
    @patch("app.main._router_agent")
    def test_chat_fails_when_conversation_memory_is_unavailable(
        self, _router_agent, _build_history
    ):
        with self.assertRaises(HTTPException) as raised:
            chat(
                ChatRequest(message="Hello"),
                AuthenticatedPrincipal(principal_id="principal-1"),
            )

        self.assertEqual(raised.exception.status_code, 503)

    @patch("app.main._chat_slots")
    def test_chat_rejects_requests_over_the_concurrency_limit(self, chat_slots):
        chat_slots.acquire.return_value = False

        with self.assertRaises(HTTPException) as raised:
            chat(
                ChatRequest(message="Hello"),
                AuthenticatedPrincipal(principal_id="principal-1"),
            )

        self.assertEqual(raised.exception.status_code, 429)
        self.assertEqual(raised.exception.headers["Retry-After"], "1")
        chat_slots.release.assert_not_called()

    @patch("app.main.build_conversation_history")
    @patch("app.main._router_agent")
    def test_chat_returns_conflict_when_conversation_is_busy(
        self, router_agent, build_history
    ):
        history = Mock()
        history.acquire_turn_lock.side_effect = ConversationBusyError(
            "This conversation is busy"
        )
        build_history.return_value = history

        with self.assertRaises(HTTPException) as raised:
            chat(
                ChatRequest(
                    message="Follow up",
                    conversation_id=UUID("9015dbdd-608a-443e-8ddb-a142817ad9df"),
                ),
                AuthenticatedPrincipal(principal_id="principal-1"),
            )

        self.assertEqual(raised.exception.status_code, 409)
        history.load_context.assert_not_called()
        history.close.assert_called_once_with()
        router_agent.return_value.invoke.assert_not_called()

    @patch("app.main._router_agent")
    def test_chat_rejects_customer_selection_without_membership(
        self, router_agent
    ):
        self.authorize_customer.side_effect = CustomerNotAuthorizedError

        with self.assertRaises(HTTPException) as raised:
            chat(
                ChatRequest(
                    message="Show my account details",
                    customer_id="someone-elses-customer",
                ),
                AuthenticatedPrincipal(principal_id="principal-1"),
            )

        self.assertEqual(raised.exception.status_code, 404)
        router_agent.assert_not_called()

    @patch("app.main._router_agent")
    def test_chat_requires_customer_selection_for_multiple_memberships(
        self, router_agent
    ):
        self.authorize_customer.side_effect = CustomerSelectionRequiredError

        with self.assertRaises(HTTPException) as raised:
            chat(
                ChatRequest(message="Show my account details"),
                AuthenticatedPrincipal(principal_id="principal-1"),
            )

        self.assertEqual(raised.exception.status_code, 400)
        router_agent.assert_not_called()

    @patch("app.main.build_conversation_history")
    @patch("app.main._router_agent")
    def test_chat_rejects_conversation_owned_by_another_principal(
        self, router_agent, build_history
    ):
        history = Mock()
        history.claim_ownership.side_effect = ConversationOwnershipError
        build_history.return_value = history

        with self.assertRaises(HTTPException) as raised:
            chat(
                ChatRequest(
                    message="Follow up",
                    conversation_id=UUID(
                        "9015dbdd-608a-443e-8ddb-a142817ad9df"
                    ),
                ),
                AuthenticatedPrincipal(principal_id="principal-1"),
            )

        self.assertEqual(raised.exception.status_code, 404)
        history.load_context.assert_not_called()
        history.close.assert_called_once_with()
        router_agent.assert_not_called()


if __name__ == "__main__":
    unittest.main()

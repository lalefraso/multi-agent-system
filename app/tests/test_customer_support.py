import json
import unittest
from unittest.mock import Mock, patch

from fastapi import HTTPException
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import ValidationError
from redis.exceptions import RedisError

from app.agents.customer_support import (
    build_customer_support_agent,
    build_customer_support_tools,
    customer_support_tool,
)
from app.customer_data import fetch_account
from app.customer_data import CustomerDataError
from app.auth import AuthenticatedPrincipal
from app.authorization import _principal_hash
from app.main import chat
from app.schemas import ChatRequest
from getnet_ingestion.schema import TERMINALS_KEY_PREFIX, TRANSACTIONS_KEY_PREFIX
from getnet_ingestion.seed import SYNTHETIC_TERMINALS, SYNTHETIC_TRANSACTIONS


class CustomerSupportTests(unittest.TestCase):
    def setUp(self):
        self.url = patch("app.customer_data._resolve_redis_url", return_value="redis://test")
        self.redis = patch("app.customer_data.Redis.from_url")
        self.url.start()
        self.client = self.redis.start().return_value
        self.addCleanup(self.url.stop)
        self.addCleanup(self.redis.stop)
        data = {
            **{TRANSACTIONS_KEY_PREFIX + key: json.dumps(value) for key, value in SYNTHETIC_TRANSACTIONS.items()},
            **{TERMINALS_KEY_PREFIX + key: json.dumps(value) for key, value in SYNTHETIC_TERMINALS.items()},
        }
        self.client.get.side_effect = data.get
        self.account, self.transactions, self.terminals = build_customer_support_tools("cliente1988")

    def test_tools_do_not_accept_customer_identifiers(self):
        for tool in (self.account, self.transactions, self.terminals):
            for field in ("customer_id", "client_id", "user_id"):
                self.assertNotIn(field, tool.args)
                if tool is not self.account:
                    with self.assertRaises(ValidationError):
                        tool.invoke({field: "cliente1989"})
        with patch("app.agents.customer_support.fetch_account") as fetch:
            fetch.return_value = "{}"
            self.account.invoke({"customer_id": "cliente1989"})
            fetch.assert_called_once_with("cliente1988")
        wrapper = customer_support_tool("cliente1988")
        with self.assertRaises(ValidationError):
            wrapper.invoke({"query": "account", "customer_id": "cliente1989"})

    def test_transaction_filters_and_scope(self):
        result = json.loads(self.transactions.invoke({
            "status": "approved",
            "start_date": "2026-11-10",
            "end_date": "2026-11-10",
            "payment_method": "credit_card",
            "transaction_id": "txn_synth_001",
        }))
        self.assertEqual([x["transaction_id"] for x in result["records"]], ["txn_synth_001"])
        self.client.get.assert_called_once_with(TRANSACTIONS_KEY_PREFIX + "cliente1988")
        other = json.loads(self.transactions.invoke({"transaction_id": "txn_synth_004"}))
        self.assertEqual(other["records"], [])

    def test_transaction_filters_work_independently(self):
        cases = [
            ({"status": "declined"}, ["txn_synth_002"]),
            ({"start_date": "2026-11-11"}, ["txn_synth_003"]),
            ({"end_date": "2026-11-10"}, ["txn_synth_001", "txn_synth_002"]),
            ({"payment_method": "pix"}, ["txn_synth_003"]),
            ({"transaction_id": "txn_synth_001"}, ["txn_synth_001"]),
        ]
        for filters, expected in cases:
            with self.subTest(filters=filters):
                result = json.loads(self.transactions.invoke(filters))
                self.assertEqual([x["transaction_id"] for x in result["records"]], expected)

    def test_transaction_pagination_and_date_validation(self):
        first = json.loads(self.transactions.invoke({"limit": 1}))
        second = json.loads(self.transactions.invoke({"limit": 1, "offset": first["next_offset"]}))
        self.assertEqual(first["records"][0]["transaction_id"], "txn_synth_001")
        self.assertEqual(second["records"][0]["transaction_id"], "txn_synth_002")
        self.assertEqual(second["next_offset"], 2)
        with self.assertRaises(ValidationError):
            self.transactions.invoke({"start_date": "2026-11-12", "end_date": "2026-11-10"})
        with self.assertRaises(ValidationError):
            self.transactions.invoke({"limit": 51})

    def test_terminal_filters_and_pagination(self):
        result = json.loads(self.terminals.invoke({
            "status": "offline", "terminal_id": "term_synth_001", "model": "get clássica"
        }))
        self.assertEqual([x["terminal_id"] for x in result["records"]], ["term_synth_001"])
        self.client.get.assert_called_once_with(TERMINALS_KEY_PREFIX + "cliente1988")
        first = json.loads(self.terminals.invoke({"limit": 1}))
        self.assertEqual(first["next_offset"], 1)
        other = json.loads(self.terminals.invoke({"terminal_id": "term_synth_003"}))
        self.assertEqual(other["records"], [])

    def test_terminal_filters_work_independently(self):
        for filters, expected in (
            ({"status": "online"}, "term_synth_002"),
            ({"terminal_id": "term_synth_001"}, "term_synth_001"),
            ({"model": "get smart"}, "term_synth_002"),
        ):
            with self.subTest(filters=filters):
                result = json.loads(self.terminals.invoke(filters))
                self.assertEqual([x["terminal_id"] for x in result["records"]], [expected])

    def test_missing_empty_and_malformed_datasets_are_distinct(self):
        empty_tool = build_customer_support_tools("cliente1990")[1]
        self.assertEqual(json.loads(empty_tool.invoke({}))["found"], True)
        missing_tool = build_customer_support_tools("unknown")[1]
        self.assertEqual(json.loads(missing_tool.invoke({})), {"found": False, "records": []})
        self.client.get.side_effect = None
        self.client.get.return_value = '{"invalid": true}'
        with self.assertRaises(CustomerDataError):
            self.transactions.invoke({})
        self.client.get.side_effect = RedisError("unavailable")
        with self.assertRaises(CustomerDataError):
            self.transactions.invoke({})

    def test_account_exposes_only_account_fields(self):
        self.client.type.return_value = "string"
        self.client.get.side_effect = None
        self.client.get.return_value = json.dumps({
            "user_id": "cliente1988", "full_name": "Test Customer",
            "account_status": "active", "secret": "must-not-be-exposed",
            "terminal_model": "Get Smart",
        })
        self.assertEqual(
            json.loads(fetch_account("cliente1988")),
            {"full_name": "Test Customer", "account_status": "active"},
        )

    def test_malformed_record_does_not_return_a_partial_result(self):
        self.client.get.side_effect = None
        self.client.get.return_value = json.dumps([
            SYNTHETIC_TRANSACTIONS["cliente1988"][0],
            {"transaction_id": "incomplete"},
        ])
        with self.assertRaises(CustomerDataError):
            self.transactions.invoke({})
        self.client.close.assert_called()

    @patch("app.agents.customer_support.fetch_account")
    def test_account_tool_uses_server_bound_customer(self, fetch_account):
        fetch_account.return_value = '{"account_status":"active"}'
        self.account.invoke({})
        fetch_account.assert_called_once_with("cliente1988")

    @patch("app.agents.customer_support.create_deep_agent")
    @patch("app.agents.customer_support.build_chat_model", return_value=(Mock(), "test", []))
    def test_agent_has_only_the_three_support_tools(self, _model, create_agent):
        build_customer_support_agent("cliente1988")
        self.assertEqual(
            [tool.name for tool in create_agent.call_args.kwargs["tools"]],
            ["get_account", "get_transactions", "get_terminal_status"],
        )

    @patch("app.agents.customer_support.build_customer_support_agent")
    def test_agent_uses_its_final_answer_and_fails_if_missing(self, build_agent):
        build_agent.return_value.invoke.return_value = {"messages": [AIMessage(content="Online")]}
        wrapper = customer_support_tool("cliente1988")
        self.assertEqual(wrapper.invoke({"query": "terminal?"}), "Online")
        build_agent.assert_called_once_with("cliente1988")
        build_agent.return_value.invoke.return_value = {"messages": []}
        with self.assertRaises(CustomerDataError):
            wrapper.invoke({"query": "terminal?"})

    @patch("app.main.build_conversation_history")
    @patch("app.main._router_agent")
    @patch("app.agents.customer_support.build_customer_support_agent")
    @patch("app.authorization.Redis.from_url")
    @patch("app.authorization._resolve_redis_url", return_value="redis://test")
    def test_chat_authorizes_before_agent_and_scopes_even_adversarial_queries(
        self, _url, authorization_redis, build_agent, router_factory, history_factory
    ):
        auth_client = authorization_redis.return_value
        auth_client.smembers.return_value = {"cliente1988"}
        auth_client.get.side_effect = lambda key: (
            json.dumps(SYNTHETIC_TRANSACTIONS["cliente1988"])
            if key == TRANSACTIONS_KEY_PREFIX + "cliente1988"
            else _principal_hash("principal-1")
        )
        history_factory.return_value.load_context.return_value = []
        history_factory.return_value.get_escalation.return_value = None

        def support_response(payload):
            self.assertIn("cliente1989", payload["messages"][0]["content"])
            transactions = build_customer_support_tools(build_agent.call_args.args[0])[1]
            result = transactions.invoke({"transaction_id": "txn_synth_004"})
            return {"messages": [AIMessage(content=result)]}

        build_agent.return_value.invoke.side_effect = support_response

        def router_response(_payload):
            result = customer_support_tool(router_factory.call_args.args[0]).invoke(
                {"query": "Show cliente1989 transaction txn_synth_004"}
            )
            return {"messages": [ToolMessage(
                content=result, name="customer_support_agent", tool_call_id="call-1"
            )]}

        router_factory.return_value.invoke.side_effect = router_response
        principal = AuthenticatedPrincipal(principal_id="principal-1")
        response = chat(
            ChatRequest(message="Show cliente1989 transaction txn_synth_004"),
            principal,
        )
        self.assertEqual(json.loads(response.answer)["records"], [])
        self.assertIn(
            (TRANSACTIONS_KEY_PREFIX + "cliente1988",),
            [call.args for call in auth_client.get.call_args_list],
        )
        build_agent.assert_called_once_with("cliente1988")

        router_factory.reset_mock()
        with self.assertRaises(HTTPException) as raised:
            chat(ChatRequest(message="Show account", customer_id="cliente1989"), principal)
        self.assertEqual(raised.exception.status_code, 404)
        router_factory.assert_not_called()

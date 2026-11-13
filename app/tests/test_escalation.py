import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

from fastapi import HTTPException
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from redis.exceptions import RedisError

from app.agents.human_escalation import human_escalation_tool
from app.auth import AuthenticatedPrincipal
from app.escalation import (
    EscalationDraft,
    EscalationError,
    EscalationOwnershipError,
    create_escalation,
    get_escalation,
)
from app.main import chat
from app.schemas import AgentName, AgentStatus, ChatRequest, EscalationReason


class FakeRedis:
    def __init__(self):
        self.data = {}

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value, nx=False, ex=None):
        if nx and key in self.data:
            return None
        self.data[key] = value
        return True

    def expire(self, key, seconds):
        return key in self.data

    def lock(self, *args, **kwargs):
        lock = Mock()
        lock.acquire.return_value = True
        return lock

    def lrange(self, *args):
        return []

    def close(self):
        pass


class ToolCallingModel(FakeMessagesListChatModel):
    calls: int = 0

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, *args, **kwargs):
        self.calls += 1
        return super()._generate(*args, **kwargs)


def tool_call(name, args):
    return AIMessage(content="", tool_calls=[
        {"name": name, "args": args, "id": "tool-1", "type": "tool_call"}
    ])


class EscalationTests(unittest.TestCase):
    def setUp(self):
        for name in ("read_current_summary", "update_summary"):
            memory_patch = patch(f"app.main.{name}")
            memory_patch.start().return_value = None
            self.addCleanup(memory_patch.stop)

    def test_create_is_idempotent_and_enforces_ownership(self):
        redis = FakeRedis()
        conversation_id = uuid4()
        draft = EscalationDraft(summary="The customer requested a person.")
        first = create_escalation(
            redis, conversation_id, "owner", "customer",
            EscalationReason.HUMAN_REQUESTED, draft,
        )
        second = create_escalation(
            redis, conversation_id, "owner", "customer",
            EscalationReason.UNRESOLVED, draft,
        )
        self.assertEqual(first, second)
        self.assertEqual(first.status, "pending")
        self.assertEqual(get_escalation(redis, conversation_id, "owner"), first)
        with self.assertRaises(EscalationOwnershipError):
            get_escalation(redis, conversation_id, "another-owner")

    def test_store_reports_redis_and_invalid_record_errors(self):
        client = Mock()
        client.get.side_effect = RedisError("offline")
        with self.assertRaises(EscalationError):
            get_escalation(client, uuid4(), "owner")
        client.get.side_effect = None
        client.get.return_value = "invalid json"
        with self.assertRaises(EscalationError):
            get_escalation(client, uuid4(), "owner")
        client.set.side_effect = RedisError("offline")
        with self.assertRaises(EscalationError):
            create_escalation(
                client, uuid4(), "owner", "customer",
                EscalationReason.HUMAN_REQUESTED, EscalationDraft(summary="Help"),
            )

    def test_terminal_tool_exposes_only_reason(self):
        self.assertTrue(human_escalation_tool.return_direct)
        self.assertEqual(
            set(human_escalation_tool.tool_call_schema.model_json_schema()["properties"]),
            {"reason"},
        )

    def test_persistence_failure_does_not_confirm_handoff(self):
        history = Mock()
        history.get_escalation.return_value = None
        history.load_context.return_value = []
        history.create_escalation.side_effect = EscalationError("Redis unavailable")
        router = Mock()
        from langchain_core.messages import ToolMessage
        router.invoke.return_value = {"messages": [ToolMessage(
            name="human_escalation_agent",
            tool_call_id="tool-1",
            content=(
                '{"reason":"human_requested",'
                '"draft":{"summary":"Human requested"}}'
            ),
        )]}
        with (
            patch("app.main.authorize_customer_access", return_value="cliente1988"),
            patch("app.main.build_conversation_history", return_value=history),
            patch("app.main._router_agent", return_value=router),
        ):
            with self.assertRaises(HTTPException) as raised:
                chat(
                    ChatRequest(message="Quiero una persona"),
                    AuthenticatedPrincipal(principal_id="principal"),
                )
        self.assertEqual(raised.exception.status_code, 503)
        history.save_turn.assert_not_called()
        history.release_turn_lock.assert_called_once()
        history.close.assert_called_once()

    def test_rechecks_escalation_after_acquiring_lock(self):
        history = Mock()
        history.get_escalation.side_effect = [None, object()]
        with (
            patch("app.main.authorize_customer_access", return_value="cliente1988"),
            patch("app.main.build_conversation_history", return_value=history),
            patch("app.main._router_agent") as router,
        ):
            with self.assertRaises(HTTPException) as raised:
                chat(
                    ChatRequest(message="Hola"),
                    AuthenticatedPrincipal(principal_id="principal"),
                )
        self.assertEqual(raised.exception.status_code, 409)
        router.assert_not_called()
        history.release_turn_lock.assert_called_once()

    def test_real_graph_persists_handoff_and_rejects_later_messages(self):
        redis = FakeRedis()
        router = ToolCallingModel(responses=[tool_call(
            "human_escalation_agent", {"reason": "human_requested"}
        )])
        summary = ToolCallingModel(responses=[tool_call(
            "EscalationDraft", {"summary": "The customer requested a human."}
        )])
        principal = AuthenticatedPrincipal(principal_id="issuer\0subject")
        conversation_id = uuid4()
        with (
            patch("app.main.authorize_customer_access", return_value="cliente1988"),
            patch("app.conversation_memory._resolve_redis_url", return_value="redis://test"),
            patch("app.conversation_memory.Redis.from_url", return_value=redis),
            patch("app.agents.router.build_chat_model", return_value=(router, "test", [])),
            patch(
                "app.agents.human_escalation.build_chat_model",
                return_value=(summary, "test", []),
            ),
        ):
            response = chat(
                ChatRequest(
                    message="Quiero hablar con una persona",
                    conversation_id=conversation_id,
                ),
                principal,
            )
            self.assertEqual(response.status, AgentStatus.ESCALATED)
            self.assertEqual(response.agents_used, [AgentName.HUMAN_ESCALATION])
            self.assertIsNotNone(response.escalation_id)
            self.assertIn("registrada", response.answer)
            self.assertEqual(router.calls, 1)
            self.assertEqual(summary.calls, 1)

            # A fresh history instance uses the same durable Redis state.
            with self.assertRaises(HTTPException) as raised:
                chat(
                    ChatRequest(message="Otra pregunta", conversation_id=conversation_id),
                    principal,
                )
            self.assertEqual(raised.exception.status_code, 409)
            self.assertEqual(router.calls, 1)
            self.assertEqual(summary.calls, 1)
            record = next(
                value for key, value in redis.data.items()
                if key.startswith("chat:escalation:conversation:")
            )
            self.assertIn("The customer requested a human.", record)
            self.assertNotIn("Otra pregunta", record)
            with self.assertRaises(HTTPException) as other:
                chat(
                    ChatRequest(message="Hello", conversation_id=conversation_id),
                    AuthenticatedPrincipal(principal_id="other"),
                )
            self.assertEqual(other.exception.status_code, 404)

    def test_tool_receives_recent_conversation_not_router_tool_calls(self):
        router = ToolCallingModel(responses=[tool_call(
            "human_escalation_agent", {"reason": "unresolved"}
        )])
        summary_agent = Mock()
        summary_agent.invoke.return_value = {
            "structured_response": EscalationDraft(summary="Unresolved issue")
        }
        from app.agents.router import build_router_agent

        with (
            patch("app.agents.router.build_chat_model", return_value=(router, "test", [])),
            patch(
                "app.agents.human_escalation.build_human_escalation_agent",
                return_value=summary_agent,
            ),
        ):
            result = build_router_agent("cliente1988").invoke({
                "messages": [
                    HumanMessage(content="My terminal is offline"),
                    HumanMessage(
                        content="Untrusted previous summary",
                        additional_kwargs={"conversation_summary": True},
                    ),
                    AIMessage(content="Please restart it."),
                    HumanMessage(content="It still does not work."),
                ]
            })
        self.assertEqual(router.calls, 1)
        messages = summary_agent.invoke.call_args.args[0]["messages"]
        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[0].content, "Handoff category: unresolved")
        self.assertTrue(human_escalation_tool.return_direct)
        self.assertEqual(result["messages"][-1].name, "human_escalation_agent")


if __name__ == "__main__":
    unittest.main()

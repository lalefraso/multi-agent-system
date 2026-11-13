import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from app.auth import AuthenticatedPrincipal
from app.main import chat
from app.schemas import ChatRequest
from app.semantic_memory import SemanticMemoryError, current_memory_scope


class ToolCallingModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def tool_call(name, args):
    return AIMessage(content="", tool_calls=[
        {"name": name, "args": args, "id": "tool-1", "type": "tool_call"}
    ])


class MemoryChatTests(unittest.TestCase):
    def setUp(self):
        self.history = Mock()
        self.history.get_escalation.return_value = None
        self.history.load_context.return_value = []
        self.principal = AuthenticatedPrincipal(principal_id="issuer\0subject")
        self.conversation_id = uuid4()
        patches = (
            patch("app.main.authorize_customer_access", return_value="customer-1"),
            patch("app.main.build_conversation_history", return_value=self.history),
        )
        for item in patches:
            item.start()
            self.addCleanup(item.stop)

    def test_real_agent_graph_searches_bound_past_conversations(self):
        router = ToolCallingModel(responses=[
            tool_call("customer_support_agent", {"query": "What did we discuss before?"}),
            AIMessage(content="Earlier you discussed a terminal connection."),
        ])
        support = ToolCallingModel(responses=[
            tool_call("search_conversation_memory", {"query": "previous issue"}),
            AIMessage(content="Earlier you discussed a terminal connection."),
        ])
        with (
            patch("app.main.read_current_summary", return_value="Recent: terminal offline."),
            patch("app.main.update_summary") as update,
            patch("app.agents.router.build_chat_model", return_value=(router, "openai:fake", [])),
            patch("app.agents.customer_support.build_chat_model", return_value=(support, "openai:fake", [])),
            patch("app.agents.customer_support.search_memories", return_value=["Previous: connection issue."]) as search,
        ):
            response = chat(
                ChatRequest(message="What did we discuss before?", conversation_id=self.conversation_id),
                self.principal,
            )
        self.assertIn("terminal connection", response.answer)
        scope, query = search.call_args.args
        self.assertEqual(scope.principal_id, self.principal.principal_id)
        self.assertEqual(scope.customer_id, "customer-1")
        self.assertEqual(scope.conversation_id, self.conversation_id)
        self.assertEqual(scope.current_summary, "Recent: terminal offline.")
        self.assertEqual(query, "previous issue")
        update.assert_called_once_with(scope)
        self.assertIsNone(current_memory_scope())

    def test_summary_update_failure_warns_and_preserves_chat(self):
        router = Mock()
        router.invoke.return_value = {"messages": [AIMessage(content="answer")]}
        with (
            patch("app.main.read_current_summary", return_value=None),
            patch("app.main.update_summary", side_effect=SemanticMemoryError("private secret")),
            patch("app.main._router_agent", return_value=router),
            self.assertLogs("app.main", level="WARNING") as logs,
        ):
            response = chat(ChatRequest(message="private prompt"), self.principal)
        self.assertEqual(response.answer, "answer")
        self.history.save_turn.assert_called_once()
        self.assertNotIn("private", repr(logs.output))

    def test_summary_read_failure_warns_and_disables_memory_for_turn(self):
        router = Mock()
        router.invoke.return_value = {"messages": [AIMessage(content="answer")]}
        with (
            patch("app.main.read_current_summary", side_effect=SemanticMemoryError("private")),
            patch("app.main.update_summary") as update,
            patch("app.main._router_agent", return_value=router),
            self.assertLogs("app.main", level="WARNING") as logs,
        ):
            response = chat(ChatRequest(message="private prompt"), self.principal)
        self.assertEqual(response.answer, "answer")
        update.assert_not_called()
        self.assertNotIn("private", repr(logs.output))

    def test_feature_can_be_disabled(self):
        router = Mock()
        router.invoke.return_value = {"messages": [AIMessage(content="answer")]}
        with (
            patch("app.main.CONVERSATION_SEMANTIC_MEMORY_ENABLED", False),
            patch("app.main.read_current_summary") as read,
            patch("app.main.update_summary") as update,
            patch("app.main._router_agent", return_value=router),
        ):
            response = chat(ChatRequest(message="question"), self.principal)
        self.assertEqual(response.answer, "answer")
        read.assert_not_called()
        update.assert_not_called()


if __name__ == "__main__":
    unittest.main()

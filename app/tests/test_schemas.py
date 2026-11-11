import unittest
from uuid import uuid4

from pydantic import ValidationError

from app.schemas import (
    AgentStatus,
    ChatRequest,
    ChatResponse,
    KnowledgeAgentInput,
)


class ChatSchemaTests(unittest.TestCase):
    def test_accepts_expected_api_payload(self):
        request = ChatRequest(
            message="When will yesterday's sales be deposited?",
        )

        self.assertEqual(request.message, "When will yesterday's sales be deposited?")

    def test_rejects_blank_and_extra_request_fields(self):
        with self.assertRaises(ValidationError):
            ChatRequest(message="  ")
        with self.assertRaises(ValidationError):
            ChatRequest(message="Help", user_id="customer-123")

    def test_tools_accept_their_required_inputs(self):
        knowledge = KnowledgeAgentInput(query="Help with a payment")
        self.assertEqual(knowledge.query, "Help with a payment")
        with self.assertRaises(ValidationError):
            KnowledgeAgentInput(query="Help", user_id="customer-123")

    def test_response_has_request_id_and_explicit_status(self):
        conversation_id = uuid4()
        response = ChatResponse(
            conversation_id=conversation_id,
            status=AgentStatus.COMPLETED,
            answer="Customer data returned.",
        )

        self.assertIsNotNone(response.request_id)
        self.assertEqual(response.conversation_id, conversation_id)
        self.assertEqual(response.status, AgentStatus.COMPLETED)
        self.assertEqual(response.agents_used, [])


if __name__ == "__main__":
    unittest.main()

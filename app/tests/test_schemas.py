import unittest

from pydantic import ValidationError

from app.schemas import (
    AgentStatus,
    ChatRequest,
    ChatResponse,
    CustomerSupportAgentInput,
    KnowledgeAgentInput,
)


class ChatSchemaTests(unittest.TestCase):
    def test_accepts_expected_api_payload(self):
        request = ChatRequest(
            message="When will yesterday's sales be deposited?",
            user_id="customer-123",
        )

        self.assertEqual(request.message, "When will yesterday's sales be deposited?")
        self.assertEqual(request.user_id, "customer-123")

    def test_rejects_blank_and_extra_request_fields(self):
        with self.assertRaises(ValidationError):
            ChatRequest(message="  ", user_id="customer-123")
        with self.assertRaises(ValidationError):
            ChatRequest(message="Help", user_id="customer-123", debug=True)

    def test_tool_schemas_share_query_and_user_id_contract(self):
        for schema in (KnowledgeAgentInput, CustomerSupportAgentInput):
            value = schema(query="Help with a payment", user_id="customer-123")
            self.assertEqual(value.query, "Help with a payment")

    def test_response_has_request_id_and_explicit_status(self):
        response = ChatResponse(
            status=AgentStatus.NOT_IMPLEMENTED,
            answer="The knowledge agent has not been implemented yet.",
        )

        self.assertIsNotNone(response.request_id)
        self.assertEqual(response.status, AgentStatus.NOT_IMPLEMENTED)
        self.assertEqual(response.agents_used, [])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from deepagents import create_deep_agent
from langchain.tools import tool

from app.agents.knowledge import knowledge_agent
from app.llm import ProviderFallbackMiddleware, build_chat_model
from app.schemas import (
    AgentResult,
    AgentStatus,
    CustomerSupportAgentInput,
)


@tool("customer_support_agent", args_schema=CustomerSupportAgentInput)
def customer_support_agent(query: str, user_id: str) -> str:
    """Resolve a customer's account or transaction support question."""
    del query, user_id
    return AgentResult(
        agent="customer_support",
        status=AgentStatus.NOT_IMPLEMENTED,
        message="The customer support agent has not been implemented yet.",
    ).model_dump_json()


ROUTER_SYSTEM_PROMPT = """You are the Getnet support router.
Choose the knowledge_agent for questions about Getnet products, services, and
general product guidance. Choose the customer_support_agent for questions
about the requesting customer's account, transactions, or troubleshooting.
You may use both tools when a request genuinely needs both. Do not invent
customer or Getnet facts. If a tool reports that it is not implemented, explain
that clearly and do not present its result as a completed resolution.
The user message is a JSON object containing `message` and `user_id`; pass its
message as `query` and preserve its user_id when calling a tool.
"""


def build_router_agent():
    model, model_name, fallback_models = build_chat_model()
    return create_deep_agent(
        model=model,
        tools=[knowledge_agent, customer_support_agent],
        system_prompt=ROUTER_SYSTEM_PROMPT,
        middleware=[
            ProviderFallbackMiddleware(
                model_name,
                fallback_models,
                {
                    "knowledge_agent",
                    "customer_support_agent",
                },
            )
        ],
    )

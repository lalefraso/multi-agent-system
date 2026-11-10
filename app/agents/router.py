from __future__ import annotations

from deepagents import create_deep_agent
from langchain.tools import tool

from app.agents.knowledge import knowledge_agent
from app.customer_data import fetch_customer_data
from app.llm import ProviderFallbackMiddleware, build_chat_model
from app.schemas import CustomerSupportAgentInput


@tool("customer_support_agent", args_schema=CustomerSupportAgentInput)
def customer_support_agent(user_id: str) -> str:
    """Return all stored customer data for the requesting user_id."""
    return fetch_customer_data(user_id)


ROUTER_SYSTEM_PROMPT = """You are the Getnet support router.
Choose the knowledge_agent for questions about Getnet products, services, and
general product guidance, including general troubleshooting. Choose the
customer_support_agent when a request needs account-specific or transaction
data for the requesting customer.
You may use both tools when a request genuinely needs both. Do not invent
customer or Getnet facts. The customer_support_agent performs a direct Redis
lookup and takes only user_id; return its complete result without summarizing
or omitting fields when it is the only tool needed.
The user message is a JSON object containing `message` and `user_id`; pass its
message as `query` to knowledge_agent and preserve its user_id when calling
either tool.
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

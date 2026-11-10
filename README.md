# Getnet Multi-Agent System

Initial API and schema for the Getnet multi-agent support system. The router is
the entry point and has two tools: a knowledge agent and a customer support
agent. The knowledge agent uses the configured Redis vector index and
allowlisted external resources. Customer support remains an explicit
`not_implemented` stub for its later implementation.

## Architecture

```text
POST /chat
  -> ChatRequest schema
  -> Router agent (LangChain)
       -> knowledge_agent tool
       -> customer_support_agent tool
  -> ChatResponse schema
```

The router is built with LangChain's `init_chat_model` and Deep Agents'
`create_deep_agent`. Its model and ordered fallback chain are configured in
[`settings.toml`](./settings.toml). The default is OpenAI, followed by Claude,
then the OpenAI-compatible Baseten endpoint. Change `llm.model` there to select
another provider/model identifier, such as `openai:gpt-5.5`. LangSmith tracing
is enabled when `LANGSMITH_API_KEY` is configured. The existing ingestion
pipeline and Redis vector index are in [`getnet_ingestion/`](./getnet_ingestion/);
this skeleton does not change or recreate that database.

Deep Agents includes general-purpose built-in tools; router middleware limits
the tools exposed to its model to `knowledge_agent` and
`customer_support_agent`. The fallback middleware logs provider attempts,
message counts, duration, and success/failure using Python logging; it does not
log prompt or response contents.

The knowledge agent has its own two-tool boundary: `vector_search` queries the
existing `idx:getnet` index with the same multilingual E5 model and `query: `
prefix used by ingestion, and `external_resource_search` uses Tavily Extract.
Add exact HTTPS page URLs to `[knowledge].allowed_urls` in `settings.toml`.
External lookup stays disabled while the list is empty; it never accepts a URL
from the user's query, passes only configured URLs to Tavily, and filters
returned source URLs against that exact list. Set `TAVILY_API_KEY` in `.env`
when enabling external lookup.

## Configuration and run

Create `.env` from [`.env.example`](./.env.example), then set the API key for
the selected primary provider, the Anthropic fallback, and LangSmith. Baseten
is initialized only if the primary and Claude fallback fail; at that point its
API key, deployed endpoint's `BASETEN_BASE_URL`, and `BASETEN_MODEL` must be
configured. Edit `settings.toml` to choose another primary model or fallback
chain. The `thinking_level` setting is passed through for Google GenAI models.

Install dependencies and run the API from the repository root:

```bash
python -m pip install -r requirements.txt
uvicorn app.main:app --reload
```

The service provides `GET /health` and `POST /chat`. Example request:

```json
{
  "message": "When will the money from yesterday's sales be deposited?",
  "user_id": "cliente1988"
}
```

The response schema is:

```json
{
  "request_id": "generated-uuid",
  "status": "not_implemented",
  "answer": "The customer support agent has not been implemented yet.",
  "agents_used": ["customer_support"]
}
```

`status` is `completed` or `not_implemented`; `agents_used` can contain
`knowledge` and/or `customer_support`. Until an agent is implemented, a routed
request returns `not_implemented` instead of implying that it retrieved or
resolved anything. Invalid requests are rejected by the schema.

## Tests

Run the app tests with:

```bash
python -m unittest discover -s app/tests -p 'test_*.py'
```

The existing ingestion tests are in `getnet_ingestion/tests/` and can be run
independently. Future agent work should add mocked orchestration tests and
integration tests against a test Redis index, without modifying production
data.

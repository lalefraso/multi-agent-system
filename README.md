# Getnet Multi-Agent System

The router is the entry point and has two tools: a knowledge agent and a direct
customer-data lookup. The knowledge agent uses the configured Redis vector
index and allowlisted external resources. Customer support is a Redis lookup,
not a separate LLM agent.

## Architecture

```text
POST /chat
  -> ChatRequest schema
  -> Redis conversation history (last 20 messages, 30-day TTL)
  -> Router agent (LangChain)
       -> knowledge_agent tool
       -> customer_support_agent tool (Redis lookup by user_id)
  -> Redis conversation history stores the user turn and final answer
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

The customer-support tool reads the complete record at
`getnet_users:customer:<user_id>` from the same Redis connection configured by
`GETNET_REDIS_URL` or `REDIS_URL`. It returns hash, string, or RedisJSON records
as JSON and reports a missing record explicitly. The user's message is not used
to filter the record. A customer-support-only response returns the lookup
output directly instead of asking another LLM to rewrite it.

Conversation memory uses `langchain-redis` `RedisChatMessageHistory` on that
same Redis deployment. Each conversation is scoped to both `user_id` and a
`conversation_id`; when omitted, the API generates one and returns it in the
response. Clients should send that ID with later turns to continue the same
conversation. Redis expires individual messages after 30 days, and only the
most recent 20 messages are included in the agent context. This history store
uses Redis JSON and Redis Search, so the Redis deployment must provide those
modules. History read/write failures return HTTP 503 rather than silently
starting a conversation without memory.

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
  "user_id": "cliente1988",
  "conversation_id": "9015dbdd-608a-443e-8ddb-a142817ad9df"
}
```

Omit `conversation_id` on the first turn; use the ID returned by the API to
continue that conversation.

The response schema is:

```json
{
  "request_id": "f9270a7c-5347-452d-910c-d70c025a24a4",
  "conversation_id": "9015dbdd-608a-443e-8ddb-a142817ad9df",
  "status": "completed",
  "answer": "{\"account_status\":\"active\",\"user_id\":\"cliente1988\"}",
  "agents_used": ["customer_support"]
}
```

`answer` contains the customer's JSON record as a string when the customer
lookup is the only tool used. A missing record is returned as
`{"found":false,"user_id":"..."}`. `agents_used` can contain `knowledge`
and/or `customer_support`. Invalid requests are rejected by the schema.
`user_id` identifies the Redis key but is not authentication; deployments must
authenticate callers and authorize access to that user before exposing this
endpoint.

## Tests

Run the app tests with:

```bash
python -m unittest discover -s app/tests -p 'test_*.py'
```

The existing ingestion tests are in `getnet_ingestion/tests/` and can be run
independently. Customer lookup tests mock Redis so they do not read production
customer data.

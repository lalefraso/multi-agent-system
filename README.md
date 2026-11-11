# Getnet Multi-Agent System

The router is the entry point and has two tools: a knowledge agent and a direct
customer-data lookup. The knowledge agent uses the configured Redis vector
index and allowlisted external resources. Customer support is a Redis lookup,
not a separate LLM agent.

## Architecture

```text
POST /chat
  -> JWT/OIDC authentication and verified principal (issuer + subject)
  -> Redis membership check for the requested customer_id
  -> ChatRequest schema (message, optional authorized customer_id, conversation_id)
  -> Redis conversation history (last 20 messages, 30-day TTL)
  -> Router agent (LangChain)
       -> knowledge_agent tool
       -> zero-argument customer_support_agent (server-scoped customer lookup)
  -> Redis conversation history stores the user turn and final answer
  -> ChatResponse schema
```

The router is built with LangChain's `init_chat_model` and Deep Agents'
`create_deep_agent`. Its model and ordered fallback chain are configured in
[`settings.toml`](./settings.toml). The default is Baseten's
`deepseek-ai/DeepSeek-V4.1-Flash`,
followed by OpenAI and Anthropic. Change `llm.model` there to select another
provider/model identifier, such as `openai:gpt-5.5`. LangSmith tracing is enabled
when `LANGSMITH_API_KEY` is configured. The existing ingestion
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
Knowledge answers are concise and grounded only in retrieved sources. Similar
product names are treated as candidates, not assumed aliases. The vector
search also runs a targeted English lookup when it sees "Get Clássica",
"Get Clasica", or "Get Classic", since Get Classic appears in the English
corpus; this retrieves a candidate without assuming the names are equivalent.
Comparisons search for each product and report when the sources do not
establish a difference.
Add exact HTTPS page URLs to `[knowledge].allowed_urls` in `settings.toml`.
External lookup stays disabled while the list is empty; it never accepts a URL
from the user's query, passes only configured URLs to Tavily, and filters
returned source URLs against that exact list. Set `TAVILY_API_KEY` in `.env`
when enabling external lookup.

The customer-support tool has no model-visible arguments. The API creates it
for the authenticated request and binds the server-authorized customer ID in
the tool's closure; the model can request a lookup but cannot choose a
customer. It reads `getnet_users:customer:<customer_id>` from the same Redis
connection configured by `GETNET_REDIS_URL` or `REDIS_URL`. Identity fields
named `user_id`, `customer_id`, or `merchant_id` are removed from the tool
result. Neither the authenticated principal nor the selected customer ID is
included in the message sent to the model.

`POST /chat` requires an OIDC access token in the `Authorization: Bearer`
header. Configure `OIDC_ISSUER`, `OIDC_AUDIENCE`, and `OIDC_JWKS_URL`; the API
validates the JWT signature (RS256), issuer, audience, expiration, and subject
against the provider's signing keys. The verified principal is the pair of
issuer and `sub`; it is not itself a customer ID. Server-side authorization
uses a Redis Set of customer IDs for each principal. Its key is
`getnet:authz:principal:<sha256(issuer + NUL + sub)>:customers`; only trusted
administrative provisioning should modify these Sets. If the principal has
one customer, the API selects it automatically. If it has multiple customers,
the request must select one with `customer_id`; the API checks membership
before creating the agent. A supplied ID is only a selection, never proof of
access. This first version assumes each existing customer record belongs to
one principal; shared customer/merchant accounts need an explicit different
business rule. Each customer ID also has a unique reverse owner key:
`getnet:authz:customer:<sha256(customer_id)>:owner`. Grant/revoke operations
update the principal Set and reverse owner atomically in Redis. A second
principal cannot be granted the same customer, and authorization checks the
reverse owner before using a customer record. Missing or invalid tokens are
rejected; an unavailable signing-key or authorization service fails closed
with HTTP 503. Health checks remain unauthenticated.

Provision or revoke a membership using a trusted administrator operation (the
command uses `REDIS_URL` or `GETNET_REDIS_URL` from `.env`):

```bash
python -m app.manage_authorization grant \
  --issuer 'https://issuer.example/' \
  --subject 'user-subject' \
  --customer-id 'customer-123'

python -m app.manage_authorization revoke \
  --issuer 'https://issuer.example/' \
  --subject 'user-subject' \
  --customer-id 'customer-123'
```

The grant is idempotent for the same principal/customer pair and fails if
another principal already owns the customer; it also removes that stale
duplicate membership from the attempted principal. Revoke only succeeds for
the current owner. Use exact issuer and subject values from the verified
identity provider. Never create memberships from an unverified request or let
the model modify them. Redis must permit `EVAL` for this atomic operation.
If memberships were previously added with direct `SADD` commands, re-grant
each legitimate principal/customer pair with this command to create the owner
index. If a customer was in multiple principal Sets, choose the one legitimate
owner; grants to the others will fail and remove those stale duplicate entries.

This initial authorization model controls access to the existing customer
records; it does not yet model merchants, merchant roles, or a merchant-to-
customer hierarchy. Add that domain and its explicit policies when those data
and business rules are defined.

The customer lookup currently sends the rest of the customer's Redis record to
the model. Before production, replace that broad response with an allowlist of
fields needed for each supported request, and enforce any required token scopes
or tenant/role policy. Also use per-identity rate limits and audit events that
avoid logging bearer tokens, prompts, or customer data; protect Redis with
network restrictions, TLS, and least-privilege credentials.

Conversation memory uses Redis lists on the same deployment. A conversation
owner record binds each `conversation_id` to a hash of the authenticated
principal and selected customer; the history key is also scoped to both. A
different principal or customer cannot read that history, even if they submit
the same conversation ID. When omitted, the API creates an ID and returns it.
Redis expires the owner record and conversation after 30 days, and only the
most recent 20 messages are included in agent context. Redis Search is required
for vector knowledge lookup; RedisJSON is required if customer records are
stored as RedisJSON. History read/write failures return HTTP 503 rather than
silently starting a conversation without memory.

The API limits concurrent chat requests per process with a non-blocking
semaphore. Set `CHAT_MAX_CONCURRENT_REQUESTS` in `.env` (default: `8`); requests
over the limit receive HTTP 429 and a `Retry-After` header. The effective
instance limit is this value multiplied by the number of Uvicorn workers. For
example, 4 workers with a limit of 8 allow up to 32 simultaneous requests on
that instance. This is a per-process limit, not a global quota across multiple
instances; keep the combined limit within the LLM provider's rate and token
quotas.

Requests for the same conversation acquire a Redis lock before loading history
and hold it through the agent response and history write. This serializes turns
across workers and replicas sharing Redis. If the lock cannot be acquired
within 5 seconds, the API returns HTTP 409 so the client can retry shortly.
Locks expire after 15 minutes if a worker terminates unexpectedly; keep the
maximum request duration below that lease time.

## Configuration and run

Create `.env` from [`.env.example`](./.env.example), then set the API key for
the selected primary provider and any fallbacks you want available. Baseten
requires its API key and deployed endpoint's `BASETEN_BASE_URL`; its model is
taken from the `baseten:<model>` identifier in `settings.toml` unless
overridden by `BASETEN_MODEL`. Edit `settings.toml` to choose another primary
model or fallback chain. The `thinking_level` setting is passed through for
Google GenAI models. Dependency ranges are maintained in `requirements.txt`;
Docker and CI install exact, hash-verified versions from `requirements.lock`.
Regenerate the lock for Python 3.12 with:

```bash
uv pip compile --python-version 3.12 --universal --generate-hashes requirements.txt --output-file requirements.lock
```

### Run with Docker

Install Docker Desktop, copy `.env.example` to `.env`, and set the API key for
your selected LLM provider. Then build and start the API with Redis Stack, which
provides Redis Search for knowledge lookup and RedisJSON for JSON customer
records:

```bash
docker compose up --build
```

The API is available at `http://localhost:8000`. `GET /health` checks Redis,
the configured vector index, and the primary LLM provider's configuration and
credentials without making a billable LLM request; it returns HTTP 503 if any
check is unavailable. The API container runs as a non-root user, and the Redis
Stack image is pinned to a version and digest. Redis data is stored in the
`redis-data` volume and survives container restarts. Stop the services with
`docker compose down`. To also erase the local Redis history and data, run
`docker compose down --volumes`.
LangSmith tracing is disabled by default for this local stack; set
`LANGSMITH_TRACING=true` in `.env` to opt in.

Continue with the two `/chat` requests below to check that the second request
recalls the phrase from the first.

Install dependencies and run the API from the repository root:

```bash
python -m pip install --require-hashes -r requirements.lock
uvicorn app.main:app --reload
```

The service provides `GET /health` and `POST /chat`. Example request:

```json
{
  "message": "When will the money from yesterday's sales be deposited?",
  "customer_id": "customer-123",
  "conversation_id": "9015dbdd-608a-443e-8ddb-a142817ad9df"
}
```

Send this body with an OIDC Bearer token. `customer_id` is optional if the
principal has access to exactly one customer. If it has access to several,
include an authorized ID to select the customer for this turn. Omit
`conversation_id` on the first turn; use the returned ID on later turns with
the same token and customer selection. The server verifies both permissions
and conversation ownership.

To test memory across two requests, start the API with a configured LLM and a
Redis deployment that supports Redis JSON and Redis Search. Send a first turn:

```bash
curl -sS http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -d '{"message":"Remember this test phrase: cobalt otter 47.","customer_id":"customer-123"}'
```

Copy the returned `conversation_id` into the next request, using the same
authenticated identity and customer selection:

```bash
curl -sS http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -d '{"message":"What test phrase did I ask you to remember?","customer_id":"customer-123","conversation_id":"PASTE-CONVERSATION-ID-HERE"}'
```

The second answer should recall `cobalt otter 47`. Use a customer ID that was
provisioned for the token's principal. If a different principal or customer
tries to reuse the conversation ID, the API returns HTTP 404.

The response schema is:

```json
{
  "request_id": "f9270a7c-5347-452d-910c-d70c025a24a4",
  "conversation_id": "9015dbdd-608a-443e-8ddb-a142817ad9df",
  "status": "completed",
  "answer": "{\"account_status\":\"active\"}",
  "agents_used": ["customer_support"]
}
```

`answer` contains the customer's JSON record as a string when the customer
lookup is the only tool used. A missing record is returned as
`{"found":false}`. `agents_used` can contain `knowledge` and/or
`customer_support`. Invalid requests are rejected by the schema; missing or
invalid authentication is rejected before the chat handler runs.

## Tests

Run the app tests with:

```bash
python -m unittest discover -s app/tests -p 'test_*.py'
python -m unittest discover -s getnet_ingestion/tests -p 'test_*.py'
```

CI installs the locked dependencies and runs both suites. Customer lookup
tests mock Redis so they do not read production customer data.

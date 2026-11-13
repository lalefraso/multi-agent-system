# Getnet Multi-Agent System

The router is the entry point and delegates to a knowledge agent and a customer
support agent. The knowledge agent uses the configured Redis vector index and
allowlisted external resources. The customer support agent uses scoped Redis
tools for account, transaction, and terminal information.

## Architecture

```text
POST /chat
  -> JWT/OIDC authentication and verified principal (issuer + subject)
  -> Redis membership check for the requested customer_id
  -> ChatRequest schema (message, optional authorized customer_id, conversation_id)
  -> Redis conversation history (last 20 messages, 30-day TTL)
  -> Router agent (LangChain)
       -> knowledge_agent tool
       -> customer_support_agent(query), scoped to the authorized customer
            -> get_account, get_transactions, get_terminal_status
       -> human_escalation_agent(reason), terminal handoff
  -> Redis conversation history stores ordinary turns;
     escalated turns store a separate handoff record
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
the tools exposed to its model to `knowledge_agent`, `customer_support_agent`,
and `human_escalation_agent`. The fallback middleware logs provider attempts,
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

### Exchange-rate references

The knowledge agent can also look for the latest published central-bank
reference rate for USD/ARS, USD/BRL, USD/CLP, USD/MXN and USD/PYG, or EUR/USD
for Portugal and Spain. These are reference rates, **not Getnet rates**. The
agent should cite the source and publication date, and say when it cannot
verify the rate. Historical rates, other currency pairs, calculated cross
rates and Getnet-applied rates are outside this feature.

Exchange-rate lookup is disabled by default, independently of
`[knowledge].allowed_urls`. A trusted administrator must verify and configure
the exact HTTPS pages that actually publish the relevant rates:

```toml
[knowledge.exchange_rate_urls]
# mexico = ["https://www.banxico.org.mx/path-to-a-verified-rate-page"]
```

The example is a placeholder, not a verified page. The supported keys and
permitted hosts are `argentina` (bcra.gob.ar), `brazil` (bcb.gov.br),
`chile` (bcentral.cl), `mexico` (banxico.org.mx), `portugal` and `spain`
(ecb.europa.eu), and `paraguay` (bcp.gov.py); the corresponding `www` host is
also allowed. Add only exact pages from the appropriate host, not arbitrary
subdomains. Portugal and Spain must each be configured if both are needed.
Tavily Extract receives only those configured URLs and a fixed country-pair
query; returned URLs must match the configured pages. User-supplied URLs and
links discovered in retrieved content are not opened. A page may be
inaccessible or omit the required rate or date; configuring it does not
guarantee a current result. The application does not independently verify the
date or accuracy of model-interpreted page content. Verify page accessibility,
content, and freshness before enabling a source, and set `TAVILY_API_KEY`.

The server binds the authorized customer ID to the customer support agent and
each of its tools; neither model can supply or change that ID. The router sends
the user's question as `query`, and the support agent uses:

- `get_account`: account fields only, from `getnet:customers:<customer_id>`;
  falls back to `getnet_users:customer:<customer_id>`.
- `get_transactions`: `getnet:transactions:<customer_id>`, with optional
  status, inclusive local `start_date`/`end_date` (YYYY-MM-DD), payment method,
  and transaction ID filters.
- `get_terminal_status`: `getnet:terminals:<customer_id>`, with optional
  status, terminal ID, and case-insensitive exact model filters.

Transaction and terminal datasets are JSON arrays in Redis. Each tool returns
at most 50 records (default 10); `offset` and `next_offset` support pagination.
`found: false` means the dataset is missing, while `found: true` with an empty
array means it exists but has no matching records. Amounts are in minor units.
The tools only expose allowlisted fields and use the same Redis connection
configured by `GETNET_REDIS_URL` or `REDIS_URL`.

`POST /chat` requires an OIDC access token in the `Authorization: Bearer`
header. Configure `OIDC_ISSUER`, `OIDC_AUDIENCE`, and `OIDC_JWKS_URL`; the API
validates the JWT signature (RS256), issuer, audience, expiration, and subject
against the provider's signing keys. The verified principal is the pair of
issuer and `sub`; it is not itself a customer ID. Server-side authorization
uses a Redis Set of customer IDs for each principal. Its key is
`chat:authz:principals:<sha256(issuer + NUL + sub)>:customers`; only trusted
administrative provisioning should modify these Sets. If the principal has
one customer, the API selects it automatically. If it has multiple customers,
the request must select one with `customer_id`; the API checks membership
before creating the agent. A supplied ID is only a selection, never proof of
access. This first version assumes each existing customer record belongs to
one principal; shared customer/merchant accounts need an explicit different
business rule. Each customer ID also has a unique reverse owner key:
`chat:authz:customers:<sha256(customer_id)>:owner`. Grant/revoke operations
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

Before production, review the allowlisted customer fields for data minimization
and enforce any required token scopes or tenant/role policy. Also use
per-identity rate limits and audit events that
avoid logging bearer tokens, prompts, or customer data; protect Redis with
network restrictions, TLS, and least-privilege credentials.

## Human escalation

The router can call `human_escalation_agent` when a user requests a person,
asks for an unavailable sensitive action, reports fraud or an unrecognized
action, makes a formal or legal complaint, or needs human help with an
unresolved issue. Routing is model-based and is not a guaranteed security
guardrail or fraud detection mechanism. A payment decline, pending payment,
or sensitive data alone does not establish fraud.

The agent summarizes the recent conversation and the handoff reason. Its
terminal tool stops the router without another model response. The server
stores a pending case before responding; it never performs the requested
financial or account action. A successful response includes
`"status": "escalated"` and `escalation_id`, and confirms registration in
Spanish without claiming that an operator is connected. The escalation record
contains the authorized customer, a hash scoped to the principal and customer,
the conversation ID, reason, summary, UTC creation time and status. Statuses
are `pending`, `taken`, and `closed`; only `pending` is created in this
increment. Records have no expiry.

An escalated conversation is terminal for the automated agents: subsequent
`POST /chat` messages for that conversation return HTTP 409 and are not stored
or passed to a model. The same check applies after history ownership expires,
and mismatched ownership returns HTTP 404. Case creation is idempotent per
conversation. The escalation itself contains the summary rather than a full
conversation snapshot; the summarizer only sees the recent conversation context.
The triggering message is not separately added to the conversation history.
There is no reply delivery or live handoff. A pending case does not mean that
a person has received or reviewed it. The operator API sets `overdue: true`
when a case has remained pending for more than one hour. An optional worker
(`python -m app.escalation_alerts`) sends a proactive Slack alert for each such
case; it must be scheduled externally, for example every five minutes with
cron:

```cron
*/5 * * * * cd /path/to/repository && /path/to/venv/bin/python -m app.escalation_alerts
```

Set `HITL_SLACK_WEBHOOK_URL` in the deployment secret store (or local `.env`)
to enable delivery. The webhook itself is bound to the Slack channel when
provisioned. `#alertas-handoff` is only a suggested channel name; it is not
created or configured by this repository. Without a configured webhook, the
worker exits with an error and does not report a successful run. Alerts include
only the case ID, age and reason, not the summary or customer identifiers.
Successful deliveries are recorded in Redis to prevent repeat alerts; failed
deliveries are retried on a later run. Delivery is at-least-once: if Slack
accepts a message but Redis fails before recording success, a later run may
send a duplicate. This alert is informational, not an SLA or promise of
operator response. If storing the case fails, `/chat` returns HTTP 503 rather
than confirming success.

### Operator API

Operators and managers authenticate with the same OIDC issuer and audience as
`/chat`, but authentication alone does not grant access. A trusted administrator
must provision each exact issuer and subject into the appropriate, separate
Redis allowlist. The default role is `operator`; use `--role manager` for
managers:

```bash
python -m app.manage_operators grant \
  --issuer 'https://issuer.example/' --subject 'operator-subject'
python -m app.manage_operators grant --role manager \
  --issuer 'https://issuer.example/' --subject 'manager-subject'
python -m app.manage_operators revoke \
  --issuer 'https://issuer.example/' --subject 'operator-subject'
python -m app.manage_operators revoke --role manager \
  --issuer 'https://issuer.example/' --subject 'manager-subject'
```

Only trusted administrators may change `chat:authz:operators` or
`chat:authz:managers`; entries are SHA-256 hashes of `issuer + NUL + subject`,
not bearer tokens. Revocation prevents subsequent requests but does not
automatically reassign cases. An unauthorized user receives HTTP 403;
unavailable authorization storage returns HTTP 503. Each identity must belong
to only one staff role; an identity present in both allowlists is denied.

With an authorized operator or manager token:

- `GET /operator/escalations?status=pending&limit=20&offset=0` lists cases
  oldest first, with `next_offset` for pagination. `status` may be `pending`,
  `taken`, or `closed` and defaults to `pending`. Operators see all pending
  cases and only their own taken/closed cases; managers can inspect all cases.
  Each list item includes `overdue`, which is true only for pending cases older
  than one hour.
- `GET /operator/escalations/{escalation_id}` returns the customer and
  conversation identifiers, summary, reason, status, timestamps and
  `assigned_to` (the hash of the assigned operator's verified identity).
  Operators can inspect pending cases and cases assigned to them; managers can
  inspect any case.
  `conversation`, `proposed_action` and `sources` are `null`: they were not
  stored and must not be interpreted as a complete conversation or action.
- `PATCH /operator/escalations/{escalation_id}/status` accepts
  `{"status":"taken"}` or `{"status":"closed"}`. Taking a pending case
  assigns the authenticated operator; only operators may use this endpoint,
  and only the assigned operator can close a case. Managers cannot close cases.
  Repeating the same successful transition by that operator is idempotent.
  Other transitions return HTTP 409. Concurrent updates are checked atomically.
  Closing changes the status only; it does not send a reply or reopen chat.
- `PATCH /operator/escalations/{escalation_id}/assignment` accepts
  `{"issuer":"https://issuer.example/","subject":"operator-subject"}` and is
  restricted to managers. It can reassign a taken case only to an identity
  provisioned in the operator allowlist; it does not close the case or contact
  the customer. The response exposes only the assigned identity hash.

Case lookup and listing scan Redis records so existing cases need no migration.
This is suitable for the current prototype, not a high-volume queue: each
request scans records, and offset pagination can skip or repeat items when
cases change between requests. A production queue needs durable indexes and
stable pagination. The one-hour `overdue` marker is not an SLA, automatic
timeout, or alert.

The summary prompt requests omission of secrets and unnecessary personal data,
but model-generated summaries are not guaranteed to be sanitized. Before
production, add validated redaction and audit, retention and deletion policies,
backups, and monitoring. The Compose Redis enables AOF with
`appendfsync everysec` on the `redis-data` volume: a crash can still lose up to
approximately one second of writes, and volume loss is not protected. Other
Redis deployments must configure their own appropriate persistence and recovery.

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
records. The local English and Mexico embedding files must exist under
`getnet_ingestion/data/`; if needed, generate them first using the commands in
[`getnet_ingestion/README.md`](./getnet_ingestion/README.md):

```bash
docker compose up --build
```

On startup, Compose runs `getnet_ingestion.seed` once before the API. It loads
the English and Mexico Getnet embeddings from `getnet_ingestion/data/` and adds
five synthetic customer records under `getnet:customers:*`. Existing
customer records are preserved. It also seeds transaction and terminal arrays
under their respective customer-scoped prefixes, including empty arrays for
customers with no example records. Each key is created only if absent, so
existing datasets are not updated on subsequent runs. These are fixed,
fictional examples, not current transactional data. The Compose services explicitly use their
local Redis container even if `.env` contains a `GETNET_REDIS_URL` for another
environment. Redis key prefixes appear as groups (not real folders) in Redis
browser tools: business data uses `getnet:*` and support state uses `chat:*`.
Existing installations must migrate before starting the new version or the
initializer; see [Redis key migration](./getnet_ingestion/README.md#redis-key-migration).
Run the same `python -m getnet_ingestion.seed` command with the production Redis
URL to initialize that environment with the identical key and index structure.

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
  "answer": "Your account is active.",
  "agents_used": ["customer_support"]
}
```

`answer` contains the customer support agent's response when it is the only
agent used. Missing records or datasets are reported by its tools rather than
being treated as an empty result. `agents_used` can contain `knowledge` and/or
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

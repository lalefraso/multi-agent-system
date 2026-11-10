# AI Hardcore Engineer - Multi-Agent Support System
```
```

## Overview

This challenge evaluates how you engineer software, grasp AI agent design, and package it into a reliable, containerized service. The goal is to build a multi-agent system whose agents cooperate to interpret user requests and produce useful responses.

## The Task: Designing Your Agent Orchestration

You will architect and implement an Agent Orchestration composed of at least **three distinct agent types** that work together to handle incoming user messages. Use Spec Driven Delevopment, and spend at most 60 min of screen time to develop the core-solution.

## Core Requirements

<img width="721" height="459" alt="Screenshot 2025-11-21 at 10 31 26" src="https://github.com/user-attachments/assets/9d7c0d41-e2fe-4fe8-ab54-3204f70d0f87" />

### 1. Agent Orchestration Architecture

Implement at least **three distinct agent types**, plus a clear communication mechanism between them (e.g., direct function calls, internal message queue, or event-driven).

- **Agent 1 — Router Agent:**
- Serves as the primary entry point for user messages.
- Analyzes each message and decides which specialized agent (or sequence of agents) should handle it, managing the workflow and data flow between agents.
- **Agent 2 — Knowledge Agent:**
- Handles queries that require information retrieval (internal/external) and generation, answering questions about Getnet's products and services grounded in the company's website [Getnet | Máquinas de cartão e soluções financeiras para o seu negócio](https://www.getnet.net/).
- Uses a Retrieval Augmented Generation (RAG) approach plus a web search tool for general-purpose questions.
- Suggested data source for the knowledge base (please, look for other sources):
- https://www.getnet.net/en
- **Agent 3 — Customer Support Agent:**
- Provides customer support by retrieving relevant user data to resolve inquiries.
- Must create at least **2 tools**.

### 2. API Endpoint

Expose an HTTP endpoint (e.g., FastAPI) that accepts `POST` requests and returns a meaningful JSON response. Expected payload:

```json
{
"message": "Your query or statement here",
"user_id": "some_user_identifier"
}
```

### 3. Dockerization

Provide a Dockerfile (and docker-compose.yml if needed) so the application builds and runs with standard Docker commands.

### 4. Testing

Describe your overall testing strategy and how you would approach comprehensive integration testing for the agent orchestration.

### 5. Language & Frameworks

Use any language (Python or Node.js/TypeScript are common), choosing suitable libraries for the API, agents, and RAG pipeline (e.g., Langchain, LlamaIndex).

## What We're Looking For

Clean, modular, maintainable code with a well-explained multi-agent design, high-quality prompts, and a functional RAG pipeline that ingests the specified URLs. We also value solid problem solving, effective tests, an easy-to-use Docker setup, and a comprehensive README.md. We also value thoughtful approaches to AI evaluation, observability, reliability, guardrails, and production monitoring.

### README.md must cover:

- How to build, configure, and run the application, and how to run the tests.
- The orchestration architecture, design choices, message workflow, RAG pipeline (ingestion → storage → retrieval → generation), and how you leveraged LLM tools.

## Submission

Provide a link to a GitHub repository containing your solution, with a comprehensive README.md. Include a video (max 15 min duration, the time to record the video does not count in the 60 min of core-development) walking the evaluators through your solution path (your reasoning, architecture decisions, and how the orchestration works).

## Bonus Challenges

Enhance the solution with one or more of the following capabilities:

- Add a fourth custom agent of your choice, such as a Human Escalation Agent or Teams Agent that requests assistance from a human operator.
- Implement guardrails to detect and handle unsafe, irrelevant, sensitive, or unsupported requests.
- Add a redirect or human handoff mechanism for situations in which the system cannot confidently resolve the user's request.
- Present an evaluation and observability strategy for measuring the quality, reliability, and operational performance of the multi-agent system.

Candidates may also describe or implement dashboards, traces, structured logs, automated evaluation datasets, regression tests, or alerting mechanisms used to monitor the system in production.

## Example Test Scenarios

```json
{ "message": "What's the difference between the Get Clássica and the Get Smart?", "user_id": "cliente1988" }
```

```json
{ "message": "What's the weather forecast in Porto Alegre tomorrow?", "user_id": "cliente1988" }
```

```json
{ "message": "When will the money from yesterday's sales be deposited?", "user_id": "cliente1988" }
```

```json
{ "message": "Do I need a bank account to receive my sales via Pix?", "user_id": "cliente1988" }
```

```json
{ "message": "My card machine won't connect to the internet, what should I do?", "user_id": "cliente1988" }
```

```json
{ "message": "How does receivables advance (antecipação) work with Getnet?", "user_id": "cliente1988" }
```

```json
{ "message": "What's the euro exchange rate today?", "user_id": "cliente1988" }
```

```json
{ "message": "My card machine is showing a transaction decline error.", "user_id": "cliente1988" }
```

```json
{ "message": "How many installments can I split a sale into with the crediário?", "user_id": "cliente1988" }
```

```json
{ "message": "Can I sell through WhatsApp using the Payment Link?", "user_id": "cliente1988" }
```

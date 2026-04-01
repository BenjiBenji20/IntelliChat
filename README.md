# IntelliChat

<div align="center">

<!-- User will paste cover image link here -->
<img src="https://github.com/user-attachments/assets/8192c144-28e1-4cd4-966b-419a054b0c78" 
     alt="IntelliChat Cover" 
     width="100%" 
     style="max-width: 1200px; border-radius: 8px;">

[![Build Status](https://img.shields.io/badge/build-passing-success?style=flat-square)](#)
[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue?style=flat-square)](#)
[![Framework](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=flat-square&logo=fastapi)](#)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square&logo=langchain&logoColor=white)](https://www.langchain.com/)
[![Supabase](https://img.shields.io/badge/Supabase-3ECF8E?style=flat-square&logo=supabase&logoColor=white)](https://supabase.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=flat-square&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Qdrant](https://img.shields.io/badge/Qdrant-FF4C4C?style=flat-square&logo=qdrant&logoColor=white)](https://qdrant.tech/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker)](https://www.docker.com/)
[![Architecture](https://img.shields.io/badge/architecture-Microservices-8A2BE2?style=flat-square)](#)
[![Deployed on GCP](https://img.shields.io/badge/deployed%20on-GCP-4285F4?style=flat-square&logo=googlecloud)](#)
[![Google Cloud Tasks](https://img.shields.io/badge/Google%20Cloud%20Tasks-4285F4?style=flat-square&logo=googlecloud)](https://cloud.google.com/tasks)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%203.0-lightgrey.svg?style=flat-square)](https://opensource.org/licenses/AGPL-3.0)

![Project Status](https://img.shields.io/badge/status-actively%20maintained-brightgreen)
![Version](https://img.shields.io/badge/version-1.0.0--beta-blue)
![Environment](https://img.shields.io/badge/environment-production--ready%20core%20%7C%20active%20feature%20expansion-orange)

</div>

**Make the development of RAG architecture easier. Good for developers, students, start ups and small businesses.**

---

## 📑 Table of Contents
- [Quick Start](#-quick-start)
- [Feature Highlights](#-feature-highlights)
- [Architecture Overview](#-architecture-overview)
- [1. Chatbot & API Key Management](#1-chatbot--api-key-management)
- [2. Storage Layer & Document Processing](#2-storage-layer--document-processing)
- [3. RAG System (Query Pipeline)](#3-rag-system-query-pipeline)
- [4. AI Behavior Studio](#4-ai-behavior-studio)
- [5. AI Knowledge Studio](#5-ai-knowledge-studio)
- [6. Caching Architecture](#6-caching-architecture)
- [7. Token Budgeting & Context Bounds](#7-token-budgeting--context-window-management)
- [8. Operational Security](#8-security)
- [9. Performance & Async Design](#9-performance--async-design)
- [10. REST API Reference](#10-api-reference)
- [11. Complete Onboarding Flow](#11-getting-started)
- [12. Technology Decisions (ADRs)](#12-technology-decisions-adrs)
- [Roadmap](#-roadmap)
- [Known Limitations](#-known--limitations)
- [Contributing & Development Setup](#-contributing--development-setup)
- [Contact & About the Author](#-contact--about-the-author)
- [License](#-license)

---

## ⚡ Quick Start

Get the backend running locally in under a minute. The system is divided into the main API and a dedicated document worker.

```bash
# 1. Clone & Set up Virtual Environments
git clone https://github.com/BenjiBenji20/intellichat.git && cd intellichat

# 2. Main API Setup
python -m venv venv_api && source venv_api/bin/activate  # Or venv_api\Scripts\activate on Windows
pip install -r requirements.txt
cp main_api.sample.env .env  # Populate your API keys

# 3. Document Worker Setup (In a separate terminal)
python -m venv venv_doc_worker && source venv_doc_worker/bin/activate
pip install -r requirements.doc_worker.txt
cp doc_worker.sample.env doc_worker.env  # Populate worker keys

# 4. Start the Services
# Terminal 1:
uvicorn main:app --reload --port 8000
# Terminal 2:
uvicorn main_doc_worker:app --reload --port 8001
```

> **Note:** Alternatively, you can use the provided Dockerfiles (`Dockerfile` for the API and `Dockerfile.doc_worker` for the worker) to instantly spin up the environments.

---

## ✨ Feature Highlights

What makes IntelliChat scale differently from a basic OpenAI wrapper?

- **"Two-Factor" Token Firewall:** Mathematically throttles RAG retrieval limits against context window budgets, preventing infinite summarization loops and context exhaustion crashes.
- **Serverless Upload Bypass:** Gen-AI backends often crash pulling large PDFs. We use GCS Signed URLs, meaning users upload 50MB files directly to Google Cloud Storage. The FastAPI server physically never handles the bytes.
- **Absolute Vector Isolation:** Qdrant collections are programmatically grouped per specific `chatbot_id`. No query can mathematically breach the vector space of another user.
- **AI Behavior Studio:** Generates, validates, and refines high-quality system prompts using a multi-LLM review loop.
- **Fully decoupled architecture:** Heavy document processing runs asynchronously in a separate worker via Cloud Tasks.
- **Self-Healing Indexing Workers:** Background chunking jobs (deployed on Cloud Run via Cloud Tasks) use strict idempotency guards. If a task fails on step 3 out of 5, the retry safely wipes dangling artifacts before re-upserting.
- **Heavy Redis caching:** Semantic query cache + config cache for lightning-fast repeated interactions.

---

## 🏗 Architecture Overview

The system is decoupled into three fault-tolerant layers. 

<img src="https://github.com/user-attachments/assets/44e9eb26-29ae-488d-a2fb-044fcd41182f" 
     alt="IntelliChat Architecture Diagram" 
     width="80%" 
     style="border-radius: 8px; align: center;">

Description: Request lifecycle splitting conversational traffic directly to the FastAPI's Main server while heavy Document Ingestion is pushed securely via Cloud Tasks to separate worker containers.

---

## CORE FEATURES

### 1. Chatbot & API Key Management

The chatbot creation flow is highly modularized, separating identity creation and knowledge configuration (LLMs and Embedding models).

- **API Key Lifecycle:** Keys are submitted raw, but *before* saving, IntelliChat tests them against the actual provider (OpenAI, Google, Cohere) using the exact specified model name. 
- **Symmetric Encryption:** If the test passes, the key is symmetrically encrypted using Fernet (`cryptography` package). The encryption key lives strictly as an environment variable and never touches the database. Encrypted tokens are stored in Supabase and decrypted *only in-memory* strictly at request time.
- **Embedding Model Swapping:** When users update their embedding model (e.g., from OpenAI to Google), IntelliChat detects dimension mis-matches. It safely deletes the old collection entirely, recreates it with the newly requested dimensions, and automatically triggers a background re-index of all documents.

### 2. Storage Layer & Document Processing

#### The Storage Layer
- **Supabase (Postgres):** Relational state, application configuration, and encrypted secrets.
- **Google Cloud Storage (GCS):** Stores raw user documents via the Signed URL bypass pattern.
- **Qdrant:** High-performance Vector DB mapped cleanly to isolated tenants.
- **Redis (Upstash):** Serverless edge caching layer drastically cutting down DB checks for frequency-heavy data like chatbot system prompts and session memory.

### The Background Worker Pipeline (`doc_worker/`)
Deployed via Google Cloud Tasks to ensure `FastAPI` handles conversational streaming natively without blocking.
- **Idempotency Guard:** Tasks ping `document_status`. If marked `indexed`, it safely skips. If re-running, it cascades deletes on partial Qdrant artifacts using the `document_id`.
- **Chunking Subsystem:** Documents pull securely from GCS and route to dedicated parsers (`TextChunker`, `PdfChunker`, `MarkdownChunker`). PDFs bypass loaders entirely, pipelining raw bytes directly through specialized chunkers to maintain structural integrity.

### 3. RAG System (Query Pipeline)

A standard message lifecycle traces smoothly toward the LLM.
1. **Auth & Guardrails:** Secrets are validated, and the `QueryGuardrail` analyzes the input.
2. **Retrieval (`retrieval_service.py`):** Embeds the query and searches the designated Qdrant collection. Utilizes Semantic Caching in Redis—exact duplicate queries instantly bypass the Embedding SDK.
3. **Prompt Assembly:** Merges: `System Prompt` + `Recent Turns (Chat Memory)` + `Conversation Summary` + `Retrieved Knowledge` + `Current Query`. 
4. **Token Tracking & Streaming:** Triggers asynchronous streaming via standard provider SDKs. Returns payload data packed with detailed context-window usage statistics, precise costs, and retrieval metadata.

### 4. AI Behavior Studio

Crafting the perfect system prompt is offloaded entirely to a 3-Model AI-Agent Chain (`behavior_studio_script.py`):
1. **The Generator (`120b parameter model`):** Ingests the user's base identity choices (Target Audience, Tone) and creates a rich persona.
2. **The Validator (`20b parameter safeguard`):** Evaluates if the output explicitly covers Identity, RAG logic, Memory bounds, Fallbacks, and System Safety Boundaries. 
3. **The Refiner:** If validation fails, the 120b model attempts up to 2 patch cycles to gracefully integrate missing constraints.

### 5. AI Knowledge Studio

- Absolute scoping controls multitenancy. `chatbot_id` creates an unbreachable wall in `Qdrant`. 
- **Deletion Cascade:** Deleting a document from your dashboard triggers a payload-filter delete in the vector-space instantly preventing ghost-context. 

## 6. Caching Architecture

IntelliChat heavily employs Upstash Serverless caching to eliminate latency.
- **System Variables Cache:** High-frequency variables (`system_prompt`, `llm_keys`) are converted to Redis Hashes.
- **Semantic Caching:** Hashes RAG queries + active filters. Repeating identical questions costs exactly 0 embedding tokens.
- **Fire-and-Forget Pushing:** The system leverages `asyncio.create_task` to push data into Redis asynchronously behind the response cycle.

## 7. Token Budgeting & Context Window Management

IntelliChat's "Token Firewall" prevents standard `502` Model Context Exhaustion crashes regardless of the LLM provider.

- **Dynamic Profiling:** Uses `tiktoken` to construct exact allocations of contexts.
- **Retrieval Budgeting (`reduce_knowledge`):** Ensures RAG vectors never claim more than a mathematically safe ~15% limit of the context window dynamically clamping `top_k`.
- **Infinite Conversational Memory Squeeze:** When conversations grow too long, an internal background LLM task asynchronously squashes the oldest half of the history into a dense 250-word user-profile summary without affecting frontend response times.

```text
Total Model Context Window (e.g., GPT-4o: 128,000)
├── System Prompt           ~450 tokens 
├── Conversation Summary    ~200 tokens   (Dynamically squashed)
├── Recent Chat Turns       ~Variable     (Capped)
├── Retrieved Knowledge     ~Variable     (Capped at max ~15%)
├── Current User Query      ~50 tokens   
└── LLM Output Reservation  2,000 tokens  (Strictly protected buffer)
```

## 8. Security

- **Authentication:** `IntelliChat-Secret` is handled via one-way SHA-256 hashing. If the database is compromised, application keys remain completely mathematically obfuscated.
- **Restricted Origins:** `DualCORSMiddleware` applies open `*` paths solely to the embeddable chat component, securely restricting Dashboard administration routing to localized deployments.
- **Rate Limiting:** Managed via Redis Sliding Window protocols scoping IP Addresses for the public widget and User UUIDs for backend administration.

## 9. Performance & Async Design

- **100% Asynchronous Data Planes:** `FastAPI` + `Fully utilizing asyncio module` + `SQLAlchemy AsyncSession` + `AsyncQdrantClient`. Zero sequential blocking network IO calls inside critical operational paths. 
- **Sub-Routine decoupling:** Logging API costs or mutating chat-state into the database uses `asyncio.create_task()`. Response payloads stream instantly to the user while background processes manage database resolution gracefully in the background.

## 10. API Reference

**Endpoint:** `POST /{project_id}/{chatbot_id}`
**Headers:** `Content-Type: application/json` | `IntelliChat-Secret: YOUR_SECRET_KEY`

**Request Payload:**
```json
{
  "query": "How do I upgrade my billing?", 
  "conversation_id": "session_1234abcd", // user provided persistent id
  "stream": true,
  "top_k": 5, 
  "filters": [] // view the retrieval_schema.py for detailed fields
}
```

## 11. Getting Started

1. **Create an Account/Project.**
2. **Create a Project:** Create a project and configure your chatbot identity.
3. **Provider Key Integration:** Input an OpenAI/Groq API key for Chat and a Cohere/Google Key to activate Embeddings. 
4. **AI Behavior Studio:** Answer the fields and let the 3-model validator pipeline construct your system prompt.
5. **Knowledge Studio:** Upload business documents in official Intellichat interface (Your browser will directly upload to secure GCS).
6. **Embed & Roll Out.** Post directly to the REST API from your specific client!

## 12. Technology Decisions (ADRs)

* **Framework: FastAPI**
  * **Why:** Necessary for highly concurrent SSE Streaming. The requirement prompted an entirely refactored async SQLAlchemy layer.
* **Architecture: GCS Signed URL Bypass**
  * **Why:** Heavy file uploads severely bottleneck REST processing. Shifting payload bytes straight to Google Storage removed memory-pressure from FastAPI instances entirely. 
* **Worker Service: Google Cloud Tasks**
  * **Why:** Removed complex broker-dependencies (RabbitMQ/Redis) allowing background workers to scale securely serverless on-demand via HTTP.
* **Vector Store: Qdrant**
  * **Why:** Direct named-collections scoped dynamically per specific Chatbot provides an unbreachable structural layout natively superior to PostgreSQL Row-Level Security masking.

---

## 🚀 Roadmap

I am aggressively iterating on the core of IntelliChat. Here is the operational priority roadmap:

- [ ] **Free Starter Tier Integration (via n8n):** Setting up automated webhook workflows to request and pipe free Groq/Google API keys cleanly upon initial project creation to smooth out the developer experience.
- [ ] **Multimodal Agent Support:** Upgrading standard generative Chat capabilities to process image-analyzation payloads within the Memory-Squeeze contexts.
- [ ] **Native WebSockets:** While HTTP SSE streaming manages standard operations, high-fidelity operations will transition natively to async `websockets`.  
- [ ] **Function Deployment Transitions:** Off-loading the chat resolution endpoint completely to an independent Cloud Run Function to eradicate REST-free-tier coldstart behavior entirely.
- [ ] **Expand provider support:** Add support for more LLM providers and embedding providers.

---

## ⚠️ Known Limitations
- The system heavily relies on `SSE` text protocols via REST. No open persistent WebSocket pathways.
- Provider integration is currently heavily focused toward Google, Groq, Anthropic, Cohere, and OpenAI.
- Free-tier Deployments face an initial ~5 second cold-start upon document Background Processing ingestions.

---

## 💻 Contributing & Development Setup

Developers who wish to contribute or self-host IntelliChat locally will need access to Upstash, Supabase, Qdrant, and Google Cloud credentials.

**Prerequisites:**
- Python `3.12+`
- Accounts for GCP (Storage & Tasks), Qdrant Cloud, and Supabase.

1. Review the `.env` configuration requirements in `.sample.env` inside both the `api/` and `doc_worker/` applications. 
2. Ensure you utilize the `Encryption Key` efficiently. Generate a secure 256-bit token.
3. Check the `Quick Start` section above to initiate dual-terminal development monitoring.

---

## 💬 Contact & About the Author

*This section details my journey architecting this RAG system. Providing developers a scalable, decoupled, multi-tenant solution required deep-dives into edge-caching architectures and system idempotency. Contact me for more project details!*

**LinkedIn:** [Benji (Imperial) Cañones](https://www.linkedin.com/in/benji-cañones) 

**GitHub:** [BenjiBenji20](https://github.com/BenjiBenji20)  

**Email:** benjicanones6@gmail.com

---

## 📄 License
This application is distributed under the **AGPL-3.0 License**. See `LICENSE` for more information.

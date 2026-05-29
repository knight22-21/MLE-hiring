# Architecture Documentation

## Overview

This agent is designed to resolve real support tickets for DevPlatform, Claude, and Visa by combining hybrid retrieval (BM25 + dense embeddings) with LLM-powered generation. The system processes support ticket conversations, retrieves relevant documentation from a knowledge base, and generates structured responses with appropriate actions.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         main.py                                │
│                    (Orchestrator / Entry Point)                │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Ticket Ingestion                             │
│  - Load tickets from CSV                                        │
│  - Parse JSON conversation history                              │
│  - Extract full text for retrieval                              │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Corpus Loading                               │
│  - Load markdown files from data/ directory                     │
│  - Chunk by markdown headers                                    │
│  - Store as Document objects                                    │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                 HybridRetriever                                 │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │   BM25 Index     │    │ Dense Embeddings│                    │
│  │  (keyword match) │    │ (semantic sim)  │                    │
│  └────────┬────────┘    └────────┬────────┘                    │
│           │                      │                              │
│           └──────────┬───────────┘                              │
│                      ▼                                          │
│              Score Fusion (50/50)                                │
│                      │                                          │
│                      ▼                                          │
│              Top-K Documents                                     │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              Safety & PII Detection                              │
│  - Regex-based PII detection (SSN, email, phone, credit card)  │
│  - Adversarial prompt injection detection                        │
│  - Redaction of sensitive information                           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              SmartGroqPool (Multi-Key API Management)           │
│  - Round-robin key selection                                    │
│  - Per-key cooldown tracking                                    │
│  - Rate limit handling with exponential backoff                  │
│  - Concurrency control (semaphore = 15)                          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              LLM Generation (Groq Llama 3.1 8B)                  │
│  - Structured JSON output via response_format                    │
│  - Tool calling schema injection                                │
│  - Document truncation (1200 chars max) to reduce TPM           │
│  - Pydantic validation with schema remapping                     │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              SupportTicketOutput (Pydantic Model)                │
│  - status, product_area, response, justification                │
│  - request_type, confidence_score, source_documents             │
│  - risk_level, pii_detected, language, actions_taken             │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              Output CSV Generation                               │
│  - Write structured output to support_tickets/output.csv       │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Ticket Ingestion (`pipeline/ingestion.py`)

**Purpose**: Load and parse support ticket data from CSV files.

**Key Functions**:
- `load_tickets_from_csv()`: Reads CSV, parses JSON issue field, extracts conversation history
- `chunk_markdown()`: Splits markdown documents by headers for better retrieval
- `load_corpus()`: Loads all markdown files from data/ directory, chunks them, creates Document objects

**Data Flow**:
```
CSV → DictReader → JSON parse → Document objects
Markdown files → Chunk by headers → Document objects
```

### 2. Hybrid Retrieval (`pipeline/retrieval.py`)

**Purpose**: Retrieve relevant documentation using both keyword and semantic search.

**Strategy**: Hybrid BM25 + Dense Embeddings (50/50 weighted fusion)

**Why Hybrid?**
- BM25 excels at exact keyword matches (e.g., specific API names, error codes)
- Dense embeddings capture semantic similarity (e.g., "authentication" ≈ "login")
- Combining both provides robustness across different query types

**Implementation**:
- BM25: `rank_bm25.BM25Okapi` with tokenization by non-alphanumeric characters
- Dense: `sentence-transformers.all-MiniLM-L6-v2` (fast, good quality)
- Fusion: Normalize both score distributions, then weighted average
- Caching: Pre-computed embeddings saved to disk with corpus hash validation

**Performance Optimizations**:
- Embedding cache: `.model_cache/corpus_embeddings.npy`
- Corpus hash: Detects when corpus changes and invalidates cache
- Model cache: `.model_cache/` for sentence-transformer weights

### 3. Safety & PII Detection (`pipeline/safety.py`)

**Purpose**: Detect and handle sensitive information and adversarial inputs.

**PII Detection**:
- Regex patterns for: SSN, credit cards, phone numbers, email addresses
- Returns: (detected_bool, redacted_text)
- Redaction format: `[PII_TYPE_REDACTED]`

**Adversarial Detection**:
- Heuristic-based phrase matching for common prompt injections
- Examples: "disregard all previous instructions", "system prompt", "dan mode"
- Fast, rule-based approach (no LLM overhead)

**Escalation Logic**:
- If adversarial detected → status="escalated", request_type="invalid", risk_level="critical"
- If PII detected → flag in output, redact before sending to LLM

### 4. LLM Generation (`pipeline/generation.py`)

**Purpose**: Generate structured responses using LLM with tool calling support.

**Model**: Groq Llama 3.1 8B Instant (chosen for speed and cost efficiency)

**API Management**: `SmartGroqPool`
- Multi-key support: `GROQ_API_KEY`, `GROQ_API_KEY_2`, etc.
- Round-robin selection with cooldown tracking
- Rate limit handling: exponential backoff with jitter
- SDK retries disabled (handled manually)
- Concurrency limit: semaphore = 15

**System Prompt**:
- Defines role as Support Triage Agent
- Enforces strict JSON output schema
- Injects tool calling schema from `internal_tools.json`
- Rules: use only provided corpus, escalate on adversarial, no PII echo

**Tool Calling**:
- Schema loaded from `data/api_specs/internal_tools.json`
- Injected into system prompt dynamically
- Pydantic `ToolCall` model with `name` → `action` remapping
- LLM outputs tool calls in `actions_taken` array

**Document Truncation**:
- Max 1200 characters per document to reduce TPM load
- Truncation marker: `... [CONTENT TRUNCATED FOR LENGTH]`
- Reduces Groq API costs and improves speed

**Error Handling**:
- Max 15 retry attempts per request
- Falls back to escalated status on failure
- Logs errors with context

### 5. Structured Output (`models/schemas.py`)

**Purpose**: Define and validate output schema using Pydantic.

**Models**:
- `ToolCall`: action (string), parameters (dict)
  - Pre-validator remaps "name" → "action" for compatibility
- `SupportTicketOutput`: Main output model with all required fields
  - `to_csv_dict()`: Converts to CSV-compatible format

**Validation**:
- Pydantic ensures type safety and required fields
- Confidence score constrained to [0.0, 1.0]
- Literal types for enums (status, request_type, risk_level)

### 6. Configuration (`config.py`)

**Purpose**: Centralized configuration and environment management.

**Key Settings**:
- Multi-key Groq API loading
- Model cache directory setup
- Embedding cache paths
- Logging configuration
- Constants: temperature=0.0, seed=42 (deterministic)

**Environment Variables**:
- `GROQ_API_KEY`, `GROQ_API_KEY_2`, etc.: Groq API keys
- Loaded from `.env` file (gitignored)

### 7. Main Pipeline (`main.py`)

**Purpose**: Orchestrate the entire ticket processing pipeline.

**Flow**:
1. Load tickets and corpus
2. Initialize HybridRetriever
3. Process tickets concurrently (semaphore = 5)
4. Write results to CSV

**Concurrency**:
- Asyncio for parallel ticket processing
- Semaphore limits concurrent LLM calls to avoid overwhelming API
- Retrieval is synchronous but fast (CPU-bound)

## Retrieval Strategy Rationale

**Why Hybrid BM25 + Dense?**

1. **Complementary Strengths**: BM25 handles exact matches (e.g., "API_KEY"), embeddings handle semantic similarity (e.g., "troubleshoot" ≈ "fix")
2. **Robustness**: If one retrieval method fails, the other can compensate
3. **Cost-Effective**: BM25 is free, embeddings are pre-computed and cached
4. **Fast**: Both methods are fast at inference time

**Why all-MiniLM-L6-v2?**
- Fast inference (important for 241 tickets)
- Good quality for general text
- Small model size (~80MB) - easy to cache
- Widely used and well-tested

**Why 50/50 Fusion?**
- Balanced approach that doesn't favor one method
- Normalization ensures both contribute equally
- Empirically works well for RAG tasks

## Safety & Adversarial Handling

**Multi-Layer Defense**:

1. **PII Detection (Regex)**: Fast, rule-based, no false positives from LLM
2. **Adversarial Detection (Heuristics)**: Catches common prompt injection patterns
3. **System Prompt Rules**: Explicitly instructs LLM to refuse adversarial requests
4. **Escalation on Detection**: Adversarial inputs are escalated, not processed

**PII Handling**:
- Detected before LLM call
- Redacted in text sent to LLM
- Flag preserved in output (`pii_detected=True`)
- LLM instructed not to echo PII

**Adversarial Handling**:
- Heuristic check before LLM call
- If detected: return escalated response immediately
- No LLM processing for adversarial inputs
- High risk level assigned

## Escalation Decision Logic

**Escalation Triggers**:
1. Adversarial/prompt injection detected
2. LLM generation fails after retries
3. No relevant documents found (implicit - LLM may escalate)
4. Low confidence score (LLM decision)

**Escalation Response**:
- `status="escalated"`
- `response=""` (empty)
- `request_type="invalid"` (for adversarial) or appropriate type
- `justification`: Clear explanation
- `risk_level="critical"` (for adversarial) or appropriate level

**Non-Escalation**:
- `status="replied"`
- `response`: Actual answer from LLM
- `request_type`: One of: product_issue, feature_request, bug
- `confidence_score`: LLM's self-assessed confidence

## Performance Optimizations

**1. Multi-Key API Pooling**
- Multiple Groq keys multiply TPM limits
- Round-robin with cooldown prevents single-key exhaustion
- Exponential backoff with jitter reduces synchronized rate limits

**2. Embedding Caching**
- Pre-computed corpus embeddings saved to disk
- Corpus hash validation ensures cache invalidation on changes
- Avoids re-computing embeddings on every run

**3. Model Caching**
- Sentence-transformer model cached locally
- HuggingFace cache directory configured
- Avoids re-downloading model weights

**4. Document Truncation**
- 1200 character limit per document
- Reduces TPM load on LLM
- Improves generation speed

**5. Concurrency Control**
- Semaphore limits concurrent LLM calls (15 in pool, 5 in main)
- Prevents overwhelming API
- Balances speed and reliability

**Target Performance**: < 180 seconds for 241 tickets

## Known Limitations

1. **Groq Rate Limits**: Even with multi-key pooling, free tier has TPM limits that may cause slowdowns
2. **PII Detection**: Regex-based only - may miss complex PII patterns or have false positives
3. **Adversarial Detection**: Heuristic-based - sophisticated attacks may bypass
4. **Retrieval Quality**: Hybrid approach may still miss relevant documents for complex queries
5. **LLM Hallucination**: LLM may generate incorrect tool calls or cite non-existent documents
6. **No External API Calls**: Tool calling is schema-only, not actual API execution
7. **Single LLM Provider**: No failover to other providers (e.g., Gemini) as per user request

## Failure Modes

1. **All Groq Keys Rate Limited**: System waits for cooldown, may timeout
2. **Embedding Cache Corruption**: Falls back to re-computation (slower but functional)
3. **Corpus Hash Mismatch**: Re-computes embeddings automatically
4. **LLM JSON Parse Error**: Falls back to escalated response
5. **Pydantic Validation Error**: Pre-validator remaps "name" → "action", but other schema mismatches may fail
6. **Network Issues**: Retry logic with exponential backoff, max 15 attempts

## Determinism Guarantees

- **LLM Temperature**: 0.0 (no randomness)
- **Random Seed**: 42 (for any random operations)
- **Model Version**: Pinned to Llama 3.1 8B
- **Embedding Model**: Pinned to all-MiniLM-L6-v2
- **Retrieval**: Deterministic scoring (no random sampling)

## Entry Point

**Primary Entry Point**: `code/main.py`

**Run Commands**:
```bash
# Process full dataset
python code/main.py

# Process sample dataset (for testing)
python code/main.py --sample
```

**Requirements**: See `code/requirements.txt`

**Environment Setup**:
1. Copy `.env.example` to `.env`
2. Add Groq API key(s): `GROQ_API_KEY=sk-...` (and optionally `GROQ_API_KEY_2`, etc.)
3. Install dependencies: `pip install -r code/requirements.txt`

## Output Format

**Output File**: `support_tickets/output.csv`

**Columns** (all required):
- issue: Original ticket conversation (JSON)
- subject: Ticket subject
- company: Company name (DevPlatform, Claude, Visa)
- response: Generated response (empty if escalated)
- product_area: Categorized topic
- status: "replied" or "escalated"
- request_type: "product_issue", "feature_request", "bug", or "invalid"
- justification: Reasoning for decision
- confidence_score: Float 0.0-1.0
- source_documents: Pipe-separated file paths
- risk_level: "low", "medium", "high", or "critical"
- pii_detected: "true" or "false"
- language: ISO 639-1 code
- actions_taken: JSON array of tool calls

## Dependencies

**Core**:
- `groq`: LLM API client
- `sentence-transformers`: Dense embeddings
- `rank-bm25`: BM25 retrieval
- `pydantic`: Schema validation
- `python-dotenv`: Environment variables

**Data Processing**:
- `numpy`: Embedding operations
- `pandas`: Not used (pure CSV handling)

## Future Enhancements (Not Implemented)

1. **Multi-Provider Failover**: Gemini API as backup (explicitly excluded per user request)
2. **Web UI Dashboard**: Real-time visualization of processing (bonus feature)
3. **Advanced PII Detection**: NER-based PII detection
4. **Re-ranking**: Cross-encoder for better retrieval quality
5. **Actual Tool Execution**: Execute API calls instead of just schema generation
6. **Confidence Calibration**: Better confidence score estimation

# MLE Hiring Challenge - Support Ticket Agent

This agent resolves real support tickets for DevPlatform, Claude, and Visa using hybrid retrieval (BM25 + dense embeddings) and LLM-powered generation.

## Setup Instructions

### 1. Prerequisites

- Python 3.9+
- Groq API key(s) (free tier available at https://groq.com/)

### 2. Installation

```bash
# Navigate to the code directory
cd code

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your Groq API key(s)
# For single key:
GROQ_API_KEY=sk-your-groq-api-key-here

# For multiple keys (recommended for better rate limits):
GROQ_API_KEY=sk-your-first-key
GROQ_API_KEY_2=sk-your-second-key
GROQ_API_KEY_3=sk-your-third-key
```

### 4. Running the Agent

```bash
# Process the full dataset (241 tickets)
python main.py

# Process the sample dataset (10 tickets, for testing)
python main.py --sample
```

### 5. Output

The agent generates a CSV file at `../support_tickets/output.csv` with the following columns:
- issue: Original ticket conversation
- subject: Ticket subject
- company: Company name
- response: Generated response
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

## Architecture

See `ARCHITECTURE.md` for detailed architecture documentation including:
- High-level system design
- Component details
- Retrieval strategy rationale
- Safety and adversarial handling
- Performance optimizations
- Known limitations

## Performance

- **Target**: < 180 seconds for 241 tickets
- **Optimizations**: Multi-key API pooling, embedding caching, document truncation, concurrency control
- **Model**: Groq Llama 3.1 8B Instant (fast, cost-effective)

## Troubleshooting

### Rate Limit Errors
If you see rate limit errors:
- Add more Groq API keys to your `.env` file (as `GROQ_API_KEY_2`, `GROQ_API_KEY_3`, etc.)
- The system automatically rotates through keys and handles cooldowns

### Slow First Run
The first run will be slower because:
- Sentence-transformer model downloads (~80MB)
- Corpus embeddings are computed and cached
- Subsequent runs use cached embeddings and are much faster

### Embedding Cache Issues
If retrieval seems incorrect:
- Delete `.model_cache/` directory to force re-computation
- The system will automatically rebuild embeddings

## Dependencies

See `requirements.txt` for full list:
- groq: LLM API client
- sentence-transformers: Dense embeddings
- rank-bm25: BM25 retrieval
- pydantic: Schema validation
- python-dotenv: Environment variables
- numpy: Embedding operations

## Project Structure

```
code/
├── main.py              # Entry point and pipeline orchestration
├── config.py            # Configuration and environment management
├── requirements.txt     # Python dependencies
├── validate_output.py   # Output validation script
├── ARCHITECTURE.md      # Architecture documentation
├── README.md            # This file
├── pipeline/
│   ├── ingestion.py     # Ticket and corpus loading
│   ├── retrieval.py     # Hybrid BM25 + dense retrieval
│   ├── generation.py    # LLM generation with tool calling
│   └── safety.py        # PII and adversarial detection
├── models/
│   └── schemas.py       # Pydantic output schemas
├── utils/               # Utility functions (empty currently)
└── .model_cache/        # Cached model weights and embeddings (auto-created)
```

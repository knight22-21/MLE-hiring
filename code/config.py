import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
CODE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TICKETS_DIR = BASE_DIR / "support_tickets"

INPUT_CSV = TICKETS_DIR / "support_tickets.csv"
OUTPUT_CSV = TICKETS_DIR / "output.csv"

# Optional: specifically for testing
SAMPLE_CSV = TICKETS_DIR / "sample_support_tickets.csv"

# ─── Caching ───────────────────────────────────────────────
# Local cache directory for sentence-transformer model weights
MODEL_CACHE_DIR = CODE_DIR / ".model_cache"
MODEL_CACHE_DIR.mkdir(exist_ok=True)

# Set HuggingFace / sentence-transformers cache env vars BEFORE any import
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(MODEL_CACHE_DIR)
os.environ["HF_HOME"] = str(MODEL_CACHE_DIR)

# Pre-computed corpus embeddings cache (numpy array on disk)
EMBEDDING_CACHE_PATH = CODE_DIR / ".model_cache" / "corpus_embeddings.npy"
# Hash file to detect if corpus changed since last cache
CORPUS_HASH_PATH = CODE_DIR / ".model_cache" / "corpus_hash.txt"

# Setup basic logging
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger("MLE_Agent")

logger = setup_logging()

# Constants
MAX_RETRIEVED_DOCS = 5
LLM_TEMPERATURE = 0.0
LLM_SEED = 42
MAX_CONCURRENT_REQUESTS = 5

# ─── Multi-Key Groq API Support ───────────────────────────
# Collects all GROQ_API_KEY, GROQ_API_KEY_2, GROQ_API_KEY_3, ... from env
def _load_groq_keys():
    keys = []
    # Primary key
    primary = os.environ.get("GROQ_API_KEY")
    if primary:
        keys.append(primary)
    # Numbered keys: GROQ_API_KEY_2, GROQ_API_KEY_3, ...
    i = 2
    while True:
        key = os.environ.get(f"GROQ_API_KEY_{i}")
        if key:
            keys.append(key)
            i += 1
        else:
            break
    return keys

GROQ_API_KEYS = _load_groq_keys()

if not GROQ_API_KEYS:
    logger.warning("No GROQ_API_KEY found in environment. LLM calls will fail.")
else:
    logger.info(f"Loaded {len(GROQ_API_KEYS)} Groq API key(s).")

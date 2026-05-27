import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TICKETS_DIR = BASE_DIR / "support_tickets"

INPUT_CSV = TICKETS_DIR / "support_tickets.csv"
OUTPUT_CSV = TICKETS_DIR / "output.csv"

# Optional: specifically for testing
SAMPLE_CSV = TICKETS_DIR / "sample_support_tickets.csv"

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
MAX_RETRIEVED_DOCS = 3
LLM_TEMPERATURE = 0.0
LLM_SEED = 42

# Ensure Groq API Key is present if using Groq
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY is not set in the environment.")

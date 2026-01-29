import os
from pathlib import Path

from dotenv import load_dotenv

if not load_dotenv():
    print("Warning: No .env file found.")

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
METADATA_FILE = BASE_DIR / "euro_ncap_metadata.csv"
CHROMA_DIR = BASE_DIR / "chroma"

COLLECTION_NAME = "euro_ncap_knowledge_base"

LLM_MODEL = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-small"
TEMPERATURE = 0

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")

if not COHERE_API_KEY:
    raise ValueError("CRITICAL: 'COHERE_API_KEY' is missing in .env.")

if not OPENAI_API_KEY:
    raise ValueError("CRITICAL: 'OPENAI_API_KEY' is missing in .env.")

if not LLAMA_CLOUD_API_KEY:
    raise ValueError("CRITICAL: 'LLAMA_CLOUD_API_KEY' is missing in .env.")

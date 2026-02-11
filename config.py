import os
from pathlib import Path

import dspy
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

if not load_dotenv(override=True):
    print("Warning: No .env file found.")

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
METADATA_FILE = BASE_DIR / "euro_ncap_metadata.csv"
CHROMA_DIR = BASE_DIR / "chroma"

COLLECTION_NAME = "euro_ncap_knowledge_base"

LLM_MODEL = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-small"
TEMPERATURE = 0.0

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")

if not COHERE_API_KEY:
    raise ValueError("CRITICAL: 'COHERE_API_KEY' is missing in .env.")

if not OPENAI_API_KEY:
    raise ValueError("CRITICAL: 'OPENAI_API_KEY' is missing in .env.")

if not LLAMA_CLOUD_API_KEY:
    raise ValueError("CRITICAL: 'LLAMA_CLOUD_API_KEY' is missing in .env.")

HTML_FILENAME = "workflow_graph.html"


def init_global_settings():
    Settings.llm = OpenAI(model=LLM_MODEL, temperature=TEMPERATURE, api_key=OPENAI_API_KEY)
    Settings.embed_model = OpenAIEmbedding(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)

    dspy_lm = dspy.LM(model=f"openai/{LLM_MODEL}", temperature=TEMPERATURE, api_key=OPENAI_API_KEY)
    dspy.configure(lm=dspy_lm)

    return dspy_lm

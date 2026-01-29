import shutil
from pathlib import Path

import chromadb
import pandas as pd
from dotenv import load_dotenv
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_parse import LlamaParse

from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    DATA_DIR,
    EMBEDDING_MODEL,
    LLM_MODEL,
    METADATA_FILE,
    TEMPERATURE,
)

load_dotenv()

Settings.llm = OpenAI(model=LLM_MODEL, temperature=TEMPERATURE)
Settings.embed_model = OpenAIEmbedding(model=EMBEDDING_MODEL)

parser = LlamaParse(result_type="markdown", split_by_page=False)  # type: ignore
node_parser = MarkdownNodeParser()

if Path(CHROMA_DIR).exists():
    shutil.rmtree(CHROMA_DIR)

chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
chroma_collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

df = pd.read_csv(METADATA_FILE)
metadata_map = {}

for _, row in df.iterrows():
    file_name = str(row["File Name"]).strip()
    metadata_map[file_name] = {
        "file_name": file_name,
        "version": str(row["Version"]).strip(),
        "start_date": int(str(row["Start Date"]).replace("-", "")),
        "end_date": int(str(row["End Date"]).replace("-", "")),
        "protocol_type": str(row["Protocol Type"]).strip(),
        "system_domain": str(row["System Domain"]).strip(),
    }

documents = []
pdf_files = list(DATA_DIR.glob("*.pdf"))

for file_path in pdf_files:
    file_name = file_path.stem.strip()
    if file_name in metadata_map:
        file_meta = metadata_map[file_name]
        file_docs = parser.load_data(str(file_path))

        for doc in file_docs:
            doc.metadata.update(file_meta)
            documents.append(doc)

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    transformations=[node_parser],
)

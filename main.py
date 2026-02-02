import asyncio

import chromadb
import phoenix as px
from dotenv import load_dotenv
from llama_index.core import Settings, StorageContext, VectorStoreIndex, set_global_handler
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore

from config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, LLM_MODEL, TEMPERATURE
from workflow import EuroNCAPWorkflow

load_dotenv()

session = px.launch_app()
set_global_handler("arize_phoenix")


async def main():
    Settings.llm = OpenAI(model=LLM_MODEL, temperature=TEMPERATURE)
    Settings.embed_model = OpenAIEmbedding(model=EMBEDDING_MODEL)

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store, storage_context=storage_context, embed_model=Settings.embed_model
    )

    workflow = EuroNCAPWorkflow(index=index, timeout=120, verbose=True)

    # query = "What is the capital of France?"
    # query = "In v1.0 test protocol, which section mentioned test scenarios?"
    # query = "In v4.3.1 test protocol, which section mentioned test scenarios?"
    # query = "What test scenarios added in v4.3.1 compared to v1.0?"
    # query = "In v4.3.1 test protocol, which test scenarios involve oncoming target?"
    query = "List test scenario changes among v1.0, 3.0.2, and 4.3.1."
    # query = "Which test protocol was used in December 2020?"
    # query = "In which test protocol is CCFtap first added?"
    # query = "What is the difference in CCRs scenario between v1.0 and 4.3.1?"

    print("Question:")
    print(query + "\n")

    response = await workflow.run(query=query)

    print("\nAnswer:")
    print(response)

    print(f"\nPlease visit Phoenix UI at: {session.url}")  # type: ignore
    input("Press Enter to exit...")


if __name__ == "__main__":
    asyncio.run(main())

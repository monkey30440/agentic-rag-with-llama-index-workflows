# Agentic RAG with LlamaIndex Workflows

This project implements an Agentic RAG system using LlamaIndex Workflows, integrating automated data ingestion with reasoning-capable workflows for efficient document QA.

## System Architecture

* **Config**: Centralizes model parameters and environment variables.
* **Ingest**: Processes data sources and builds Chroma vector indices.
* **Workflow**: Defines Agentic RAG execution logic and state transitions.
* **Main**: System entry point.

## Quick Start

### 1. Prerequisites

Install `uv` package manager:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

```

### 2. Installation

Clone the repository and sync the environment:

```bash
git clone <repository-url>
cd agentic-rag-with-llama-index-workflows
uv sync

```

### 3. Configuration

Rename `.env.example` to `.env` and populate the API key:

```bash
cp .env.example .env

```

Edit `.env`:

```text
OPENAI_API_KEY=your_api_key_here

```

### 4. Data Ingestion

Place PDF or CSV files in the `data/` directory and run the indexing script:

```bash
uv run ingest.py

```

### 5. Execution

Run the main program to initiate the Agent:

```bash
uv run main.py

```

## File Description

* `workflow.py`: Core logic defining the event-driven RAG process.
* `ingest.py`: Data processing and Chroma vector database persistence.
* `config.py`: Configuration for LLM, Embeddings, and global settings.
* `pyproject.toml`: Dependency and version declarations.

## Notes

* Do not commit `.env` files to public repositories.
* Chroma is used as the default vector database; data persists in the `chroma/` directory.
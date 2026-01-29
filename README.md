## README.md

# Agentic RAG with LlamaIndex Workflows

這是一個基於 LlamaIndex Workflows 構建的 Agentic RAG 系統。本專案展示了如何整合自動化資料攝取（Ingestion）與具備推理能力的工作流處理，以實現高效的文件問答系統。

---

## 系統架構

1. **Config**: 統一管理模型參數與環境變數。
2. **Ingest**: 負責處理數據源並建立 Chroma 向量索引。
3. **Workflow**: 定義 Agentic RAG 的執行邏輯與狀態轉移。
4. **Main**: 系統啟動入口。

---

## 快速開始

### 1. 環境準備

本專案使用 `uv` 作為套件管理工具。請先安裝 `uv`：

```bash
pip install uv

```

### 2. 安裝依賴

複製儲存庫並同步環境：

```bash
git clone <repository-url>
cd agentic-rag-with-llama-index-workflows
uv sync

```

### 3. 配置環境變數

將 `.env.example` 重新命名為 `.env` 並填入您的 API Key：

```bash
cp .env.example .env

```

編輯 `.env`：

```text
OPENAI_API_KEY=your_api_key_here

```

### 4. 資料攝取 (Ingestion)

將您的 PDF 或 CSV 文件放入 `data/` 資料夾，執行以下指令建立索引：

```bash
python ingest.py

```

### 5. 啟動專案

執行主程式開始與 Agent 對話：

```bash
python main.py

```

---

## 檔案說明

* `workflow.py`: 核心邏輯，定義事件驅動的 RAG 流程。
* `ingest.py`: 數據處理與向量數據庫（Chroma）持久化。
* `config.py`: 模型（LLM）、嵌入模型（Embedding）與全局配置。
* `pyproject.toml`: 專案依賴與版本宣告。

---

## 注意事項

* 請勿將 `.env` 檔案上傳至任何公開儲存庫。
* 預設使用 Chroma 作為向量數據庫，數據存儲於 `chroma/` 目錄下。
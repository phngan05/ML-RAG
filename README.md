# RAG-based Machine Learning QA System

## Introduction

This project implements a **Retrieval-Augmented Generation (RAG)** system designed to answer **Machine Learning-related questions** using a knowledge base built from:

* **PDF documents**
* **Python (.py) files**
* **Jupyter Notebook (.ipynb) files**


The system integrates advanced techniques such as **query translation**, **query routing**, and **Self-RAG** to improve answer accuracy, reduce hallucination, and enhance usefulness.

---

## System Architecture

### 1. Data Ingestion

* **PDF Processing**:

  * Uses `PyMuPDF4LLM` (`langchain-pymupdf4llm`) to convert PDF content into Markdown format, with `pypdf` fallback for malformed PDFs.
  * Each page is classified as text or code; code-heavy pages are stored in the code vector database.

* **Python Files (.py)**:

  * Directly processed and stored in the code vector database.

* **Jupyter Notebook Files (.ipynb)**:

  * If a notebook does not contain code cells, store text in markdown cells in the text vector database.
---

### 2. Chunking

* Uses `RecursiveTextSplitter` to split documents into manageable chunks for embedding and retrieval.

---

### 3. Embedding

* Model: `text-embedding-3-small` (OpenAI, 1536 dimensions) via `langchain-openai`
* Converts both text and code into vector embeddings.
* The two Pinecone indexes must therefore be 1536-dimensional; re-ingest the corpus after switching embedding models.

---

### 4. Vector Database

* Uses **Pinecone**
* Two separate indexes:

  * `text-index`: for theoretical content
  * `code-index`: for source code

---

### 5. LLM

* Provider: **OpenAI** (via `langchain-openai`)
* Model: `gpt-4o-mini`

---

## Query Processing Pipeline

### 1. Query Translation

The LLM classifies the incoming query and applies one of the following strategies:

* **No Translation**: Use the original query
* **RAG-Fusion**: Generate multiple query variations
* **Step-back**: Generalize the query
* **HyDE**: Generate a hypothetical answer for retrieval

---

### 2. Query Routing

* Automatically selects the appropriate index:

  * Conceptual/theoretical queries → `text-index`
  * Code-related queries → `code-index`

---

## Self-RAG Mechanism

An iterative refinement loop ensures high-quality responses:

1. **Retrieve** relevant documents
2. **Evaluate Relevance**

   * If not relevant → return to step 1 using **Step-back query**
3. **Generate Answer**
4. **Check Hallucination**

   * If hallucinated → return to step 1
5. **Evaluate Usefulness**

   * If not useful → return to step 1 using **RAG-Fusion**
6. **Return Final Answer**

---

## Project Structure

All Python packages live under `src/` and are imported from the project root as
`src.<package>.<module>` (for example `from src.core.config import RAGConfig`).
The `app/` package contains the Streamlit layer, which depends on `src/`.

```text
ML-RAG/
│── src/
│   ├── __init__.py
│   ├── main.py                 # MLKnowledgeRAG facade + CLI entry point
│   ├── celery_app.py           # Redis broker/backend and queue configuration
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py           # RAGConfig, LLMConfig, ChunkingConfig, enums
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── document_loader.py  # PDF / .py / .ipynb loading and type tagging
│   │   ├── chunkers.py         # adaptive text + code chunking
│   │   ├── pipeline.py         # load → chunk → embed → upsert
│   │   ├── status_store.py     # persistent SQLite job status
│   │   └── tasks.py            # idempotent Celery document task
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── query_translator.py # no-op / RAG-Fusion / Step-back / HyDE
│   │   ├── vector_stores.py    # Pinecone text-index and code-index
│   │   ├── router.py           # picks the index for a query
│   │   └── retriever.py        # translation + routing + retrieval + RRF
│   │
│   ├── generation/
│   │   ├── __init__.py
│   │   └── self_rag.py         # retrieve → grade → generate → verify loop
│   │
│   └── utils/
│       ├── __init__.py
│       └── helpers.py          # logging, dedupe, RRF, file moves
│
│── app/
│   ├── __init__.py
│   ├── backend.py              # service layer used by the Streamlit UI
│   └── streamlit_app.py        # Streamlit frontend
│
│── skills/
│   └── SKILL.md
│
│── data/                       # input PDF / .py / .ipynb documents
│── processed_data/             # documents are moved here after ingestion
│
│── .env.example
│── requirements.txt            # minimal Streamlit runtime
│── requirements.dev            # runtime + ingestion + Celery (local/Docker)
│── README.md
```

Because every module uses absolute `src.*` imports, commands must be run from
the project root (not from inside `src/`).

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/phngan05/ML-RAG
cd ML-RAG
```

---

### 2. Set up virtual environment

Install `requirements-dev.txt` for local development: it includes `requirements.txt`
plus document ingestion, PDF parsing and Celery. `requirements.txt` contains only
the Streamlit query runtime and is what Streamlit Cloud installs automatically.

#### Windows (PowerShell)

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

#### Windows (CMD)

```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.dev
```

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.dev
```

---

### 3. Set up environment variables

#### Windows (PowerShell)

```powershell
New-Item .env
```

#### Windows (CMD)

```cmd
type nul > .env
```

#### macOS / Linux

```bash
touch .env
```

Add the following variables to `.env`:

```
OPENAI_API_KEY=your_api_key
PINECONE_API_KEY=your_api_key
```

---

### 4. Add documents

#### Windows

```cmd
mkdir data
```

#### macOS / Linux

```bash
mkdir -p data
```

* Create a `data/` directory in the project root
* Add your PDF, `.py` and `.ipynb` files into this folder

---

### 5. Run the program

Run from the project root so the `src` package can be imported:

#### Windows

```cmd
python -m src.main
```

#### macOS / Linux

```bash
python3 -m src.main
```

> `python src/main.py` will fail because the intra-project imports use the
> `src.*` package paths; always use the `-m` form.

---

### 6. Run the Streamlit UI

The Streamlit frontend wraps the same Self-RAG pipeline behind a web UI: `app/backend.py` is the service layer and `app/streamlit_app.py` is the frontend.

#### Windows

```cmd
streamlit run app/streamlit_app.py
```

#### macOS / Linux

```bash
streamlit run app/streamlit_app.py
```

Configuration (API keys, chat model, temperature, retrieval settings and query‑translation method) is read from `.env`/environment variables; the UI is focused on chat, retrieved sources, and the Self‑RAG grading trace. Document ingestion runs separately through the Celery worker.

### Run with Docker Compose

Create `.env` from `.env.example` and set `OPENAI_API_KEY` and
`PINECONE_API_KEY`. The Compose stack starts Redis, the Streamlit app, and a
dedicated Celery ingestion worker. The `data/` and `processed_data/` folders
are shared with the containers; the SQLite job database persists under
`processed_data/`.

```bash
docker compose up --build
```

Open <http://localhost:8501>. Stop the services with `docker compose down`.
Redis data and the SQLite job database are persisted; `docker compose down -v`
also removes the Redis volume.

### Run the worker locally

For running the app/worker outside Compose, set `CELERY_BROKER_URL`,
`CELERY_RESULT_BACKEND` and `INGESTION_STATUS_DB` in `.env` (defaults are
listed in `.env.example`). Start Redis, then run a worker from the project root:

```bash
celery -A src.celery_app:celery_app worker -Q ingestion -l INFO
```

On Windows, run the worker under WSL/Linux or use Celery's solo pool:

```powershell
celery -A src.celery_app:celery_app worker -Q ingestion -l INFO --pool=solo
```

The worker writes job state to the configured SQLite database, processes chunks
in batches, and removes stale vectors for a document before retry-safe upserts.
Files are moved to the processed folder only after successful ingestion; failed
files remain in the input folder. Worker logs include per-document progress.

---

## Key Features

* Supports both **text and code retrieval**
* Intelligent **query transformation strategies**
* **Self-RAG loop** for:

  * Reducing hallucination
  * Improving answer relevance
  * Enhancing usefulness
* Interactive **Streamlit** UI for chatting with the knowledge base, ingesting documents, and inspecting retrieved sources


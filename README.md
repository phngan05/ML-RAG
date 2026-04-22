# 📚 RAG-based Machine Learning QA System

## 🚀 Introduction

This project implements a **Retrieval-Augmented Generation (RAG)** system designed to answer **Machine Learning-related questions** using a knowledge base built from:

* **PDF documents**
* **Python (.py) files**
* **Jupyter Notebook (.ipynb) files**


The system integrates advanced techniques such as **query translation**, **query routing**, and **Self-RAG** to improve answer accuracy, reduce hallucination, and enhance usefulness.

---

## 🏗️ System Architecture

### 1. Data Ingestion

* **PDF Processing**:

  * Uses `OpenDataLoader_PDF` to convert PDF content into Markdown format.
  * If a PDF contains code, the code is extracted and stored separately in the code vector database.

* **Python Files (.py)**:

  * Directly processed and stored in the code vector database.

* **Jupyter Notebook Files (.ipynb)**:

  * If a notebook does not contain code cells, store text in markdown cells in the text vector database.
---

### 2. Chunking

* Uses `RecursiveTextSplitter` to split documents into manageable chunks for embedding and retrieval.

---

### 3. Embedding

* Model: `all-MiniLM-L6-v2`
* Converts both text and code into vector embeddings.

---

### 4. Vector Database

* Uses **Pinecone**
* Two separate indexes:

  * `text-index`: for theoretical content
  * `code-index`: for source code

---

### 5. LLM

* Provider: **Groq**
* Model: `llama-3.3-70b-versatile`

---

## 🔍 Query Processing Pipeline

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

## 🔁 Self-RAG Mechanism

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

## 📂 Suggested Project Structure

```id="f1d8k3"
project/
│── backend/
│   ├── main.py
│   ├── rag/
│   ├── retriever/
│   ├── self_rag/
│   └── utils/
│
│── frontend/
│   ├── app/
│   ├── components/
│   └── services/
│
│── data/
│   ├── pdf/
│   └── code/
│
│── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/phngan05/ML-RAG
cd project
```

---

### 2. Set up virtual environment

#### Windows (PowerShell)

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

#### Windows (CMD)

```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
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
GROQ_API_KEY=your_api_key
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

#### Windows

```cmd
python main.py
```

#### macOS / Linux

```bash
python3 main.py
```

---


## 📌 Key Features

* Supports both **text and code retrieval**
* Intelligent **query transformation strategies**
* **Self-RAG loop** for:

  * Reducing hallucination
  * Improving answer relevance
  * Enhancing usefulness

---

## 🌐 Live Demo
Website: [MLKC](https://mlkc.vercel.app/)

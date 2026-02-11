# PDF RAG Chatbot — Chat with Your Documents

An **end-to-end Retrieval-Augmented Generation (RAG) application** that allows users to upload PDFs and ask natural-language questions.
The system retrieves relevant document chunks using **FAISS vector search** and generates accurate answers using **Google Gemini LLM**.

Built with **Streamlit + LangChain + HuggingFace Embeddings** for an interactive, production-style GenAI demo.

---

# Features

* Upload multiple PDF files
* Intelligent text chunking
* Semantic embeddings using Sentence-Transformers
* FAISS vector similarity search
* LLM-powered question answering with Gemini
* Interactive chat interface (Streamlit)
* Rate-limit retry handling
* Conversation history tracking
* Persistent vector database
* Safe handling of empty context / missing answers

---

# Architecture Overview

```
PDF Upload
   ↓
Text Extraction (PyPDF2)
   ↓
Chunking (RecursiveCharacterTextSplitter)
   ↓
Embeddings (HuggingFace MiniLM)
   ↓
FAISS Vector Store
   ↓
Similarity Retrieval
   ↓
Gemini LLM (LangChain Chain)
   ↓
Answer in Streamlit Chat UI
```

---

# Tech Stack

### Core

* **Python**
* **Streamlit**
* **LangChain**

### AI / NLP

* **Google Gemini (gemini-2.0-flash)**
* **Sentence-Transformers (all-MiniLM-L6-v2)**

### Vector Search

* **FAISS**

### Utilities

* PyPDF2
* Hashlib
* Datetime
* OS / Shutil

---

# Project Structure

```
PDF_RAG_Chatbot/
│
├── app.py                # Main Streamlit application
├── faiss_index/          # Stored vector database
├── requirements.txt
└── README.md
```

---

# Installation

## Clone the repository

```bash
git clone https://github.com/<your-username>/pdf-rag-chatbot.git
cd pdf-rag-chatbot
```

## Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Mac/Linux
```

## 3Install dependencies

```bash
pip install -r requirements.txt
```

---

# Setup API Key

Get a **Google Gemini API key** and provide it in the Streamlit sidebar when running the app.

---

# Run the Application

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal.

---

# How It Works

1. Upload one or more PDFs.
2. Click **Submit** to:

   * Extract text
   * Split into chunks
   * Generate embeddings
   * Store in FAISS
3. Ask questions in natural language.
4. The system:

   * Retrieves relevant chunks
   * Sends context to Gemini
   * Generates an accurate answer

If no answer exists in the document, the bot clearly states it.

---

# Edge-Case Handling

* Handles **rate limits** with exponential retry
* Prevents querying before **embedding is ready**
* Safely manages **empty retrieval results**
* Supports **version-safe LangChain outputs**

---



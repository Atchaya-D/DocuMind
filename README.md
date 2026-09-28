# 🧠 DocuMind — Multi-Modal Document Intelligence Platform

> An intelligent document question-answering platform that uses Retrieval-Augmented Generation (RAG) to search, analyze, and understand complex PDF documents.

DocuMind enables users to upload documents, ask natural-language questions, retrieve relevant information, and receive grounded answers with document-level source references.

The system combines semantic search, vector retrieval, local LLM inference, table extraction, and visual document retrieval into a unified document intelligence workflow.

---

## ✨ Features

- 📄 **PDF Document Processing**
  - Extracts text from PDF documents
  - Preserves page-level metadata
  - Supports large multi-page documents

- 🔎 **Semantic Document Search**
  - Converts document chunks into vector embeddings
  - Uses FAISS for efficient similarity search
  - Retrieves the most relevant document sections

- 🤖 **Local RAG Question Answering**
  - Uses Retrieval-Augmented Generation
  - Runs locally using Llama 3.2 3B through Ollama
  - Answers questions using retrieved document context
  - Reduces unsupported or hallucinated responses

- 📊 **Table Extraction & Analysis**
  - Detects tables inside PDF pages
  - Extracts structured table information
  - Supports table-based information retrieval

- 🖼️ **Visual Document Retrieval**
  - Extracts images and figures from PDFs
  - Stores page and image metadata
  - Enables retrieval of document figures and charts

- 📚 **Source-Grounded Responses**
  - Displays the source document
  - Provides page-level references
  - Allows users to trace answers back to the document

- 💬 **Interactive Chat Interface**
  - Conversational document assistant
  - Chat-based question answering
  - Easy-to-use Streamlit interface

- 🔒 **Local AI Processing**
  - Uses local LLM inference
  - No external LLM API is required for the core RAG pipeline
  - Documents can remain on the local machine
  Streamlit link: https://atchaya-d-documind-app-qzpvyf.streamlit.app/

---

## 🏗️ System Architecture

```text
                    ┌──────────────────────┐
                    │       User           │
                    │  Upload PDF / Query  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Streamlit UI       │
                    │  Document Assistant   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   PDF Processing     │
                    │      PyPDFLoader      │
                    └──────────┬───────────┘
                               │
                ┌──────────────┼──────────────┐
                ▼              ▼              ▼
          ┌──────────┐   ┌──────────┐   ┌──────────┐
          │   Text   │   │  Tables  │   │  Images  │
          └────┬─────┘   └────┬─────┘   └────┬─────┘
               │              │              │
               ▼              ▼              ▼
          ┌──────────┐   ┌──────────┐   ┌──────────┐
          │ Chunking │   │  Table   │   │  Image   │
          │          │   │Extraction│   │Extraction│
          └────┬─────┘   └──────────┘   └──────────┘
               │
               ▼
       ┌──────────────────┐
       │ MiniLM Embeddings│
       └────────┬─────────┘
                │
                ▼
       ┌──────────────────┐
       │   FAISS Vector   │
       │      Store       │
       └────────┬─────────┘
                │
          User Question
                │
                ▼
       ┌──────────────────┐
       │ Similarity Search│
       └────────┬─────────┘
                │
                ▼
       ┌──────────────────┐
       │ Retrieved Context│
       └────────┬─────────┘
                │
                ▼
       ┌──────────────────┐
       │   Llama 3.2 3B   │
       │      Ollama      │
       └────────┬─────────┘
                │
                ▼
       ┌──────────────────┐
       │ Answer + Sources │
       └──────────────────┘

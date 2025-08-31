# 🧠 RAG System using OpenAI + LangChain

A Retrieval-Augmented Generation (RAG) system built using **OpenAI** and **LangChain**.  
This application enables intelligent question-answering over **databases** and **PDF documents** by combining the power of LLMs with efficient retrieval mechanisms.

---

## 🚀 Features

- 🔹 **RAG-based Question Answering** – Combines OpenAI's LLM with LangChain's retrieval pipeline.
- 🔹 **Multi-source Support** – Query data from:
  - **Databases** (posgress)
  - **PDF Documents**
- 🔹 **Contextual Responses** – Answers are generated based on retrieved chunks, improving accuracy.
- 🔹 **Embeddings + Vector Search** – Uses embeddings to find relevant information efficiently.
- 🔹 **Scalable & Modular** – Easily extendable to support other file formats or data sources.

---

## 🏗️ Architecture Overview

```plaintext
          ┌───────────────────┐
          │  User Query       │
          └───────┬───────────┘
                  │
        ┌─────────▼───────────┐
        │    Retriever        │  ← (LangChain Retriever)
        └─────────┬───────────┘
                  │
      ┌───────────┴────────────┐
      │   Data Sources         │
      │   • Database          │
      │   • PDF Files        │
      └───────────┬────────────┘
                  │
        ┌─────────▼───────────┐
        │  OpenAI LLM        │  ← (Generates final answer)
        └─────────┬───────────┘
                  │
          ┌───────▼───────────┐
          │   Final Answer    │
          └───────────────────┘


⚙️ Tech Stack

OpenAI API → For generating natural language answers

LangChain → For retrieval, context management, and prompt chaining

FAISS / ChromaDB → For vector-based similarity search

PyPDF2 / LangChain PDF Loader → For extracting and chunking PDF content

SQLAlchemy → For database connection and queries

Python 3.10+



# Clone the repository
https://github.com/RohitGanesh7/RagSystem.git
cd RagSystem

# Create a virtual environment
python -m venv venv
source venv/bin/activate   # For Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt


🔑 Environment Variables

Create a .env file in the root directory:
OPENAI_API_KEY=your_openai_api_key
DATABASE_URL=your_database_connection_string
VECTOR_DB=faiss  # or chromadb


📄 License

This project is licensed under the MIT License
.

🙌 Acknowledgments

OpenAI

LangChain

FAISS
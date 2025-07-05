# 📊 Intelligent Complaint Analysis with RAG-Powered Chatbot

This project builds a Retrieval-Augmented Generation (RAG) pipeline to help CrediTrust Financial analyze and search customer complaints more efficiently. It uses semantic search, embeddings, and a local vector database to power an internal chatbot for complaint understanding.

---

## 🧠 Problem Statement

CrediTrust receives thousands of unstructured complaint narratives. Product managers like Asha must manually review complaints to understand issues, trends, and product feedback. This process is time-consuming and inefficient.

**Goal:** Build a system that allows employees to ask questions like  
> "Why are people unhappy with savings accounts?"  
and get an intelligent, grounded summary of relevant customer complaints.

---

## ✅ Project Tasks

### ✅ Task 1: Data Exploration & Preprocessing

We performed EDA and preprocessing on complaint narratives from over 9 million records. Key steps:

- **Filtered** for 5 product categories relevant to CrediTrust:
  - Credit card
  - Personal loan
  - Buy Now, Pay Later *(if present)*
  - Savings account
  - Money transfers

- **Removed** complaints without a narrative.
- **Cleaned** the text by:
  - Lowercasing
  - Removing special characters
  - Stripping boilerplate phrases like _"I am writing to file a complaint"_
  - Normalizing whitespace

#### 📈 Insights:
- Majority of complaints were short; ~50% had fewer than 50 words.
- A large portion of complaints had missing or irrelevant filler content.
- Cleaned data saved to: `data/filtered_complaints.csv`

---

### ✅ Task 2: Text Chunking, Embedding, and Indexing

#### 🔹 Why Chunking?
Full narratives vary in length and may exceed embedding model input limits. We split each narrative into smaller overlapping chunks to preserve context.

#### 🔹 Chunking Strategy
Used `LangChain`’s `RecursiveCharacterTextSplitter` with:

- `chunk_size = 500`
- `chunk_overlap = 100`

We tested multiple configurations:

| Chunk Size | Overlap | Total Chunks | Avg Length |
|------------|---------|---------------|------------|
| 300        | 50      | 2,143,457     | 269 chars  |
| 500        | 100     | 1,381,982     | 427 chars  |
| 700        | 150     | 1,034,331     | 566 chars  |

We chose **500/100** for the best balance of context and efficiency.

#### 🔹 Embedding Model

We selected `sentence-transformers/all-MiniLM-L6-v2`, a lightweight and high-performing model for semantic similarity. It's ideal for sentence-level search and works well with CPUs.

#### 🔹 Indexing with FAISS

- Converted each chunk into a 384-dimensional vector using MiniLM.
- Stored vectors in a **FAISS** index.
- Saved accompanying metadata (`complaint_id`, `product`) for traceability.

✅ Vector store saved to: `vector_store/`

---

## 📂 Project Structure

project-root/
├── data/
│ ├── raw_complaints.parquet
│ ├── filtered_complaints.csv
│ └── chunked_complaints_500_100.csv
├── vector_store/
│ └── faiss_index/
├── notebooks/
│ ├── 01_eda_preprocessing.ipynb
│ ├── 02_chunking.ipynb
│ └── 03_embed_index.ipynb


---

## 🚀 Next Step (Task 3)

Build a **RAG pipeline** to:
- Use vector store to retrieve relevant chunks
- Pass those chunks into an LLM (e.g., Mistral, GPT-3.5)
- Generate grounded answers to product manager questions

---

## 📌 Requirements

```bash
pip install pandas tqdm sentence-transformers langchain faiss-cpu
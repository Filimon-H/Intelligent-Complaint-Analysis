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
## 🧠 Task 3: RAG Core Logic and Evaluation

### 🎯 Objective

To build a modular **Retrieval-Augmented Generation (RAG)** system that accurately answers user questions about customer complaints by combining semantic retrieval and generative AI.

---

### 🏗️ Key Components

- **Retriever Module**  
  Embeds the user’s question using the same `all-MiniLM-L6-v2` model from Task 2 and retrieves the top 5 most relevant chunks using FAISS.

- **Prompt Template**

  ```text
  You are a financial analyst assistant for CrediTrust.
  Your task is to answer questions about customer complaints.
  Use the following retrieved complaint excerpts to formulate your answer.
  If the context doesn't contain the answer, state that you don't have enough information.

  Context:
  {context}

  Question: {question}
  Answer:
This prompt helps reduce hallucination and ensures the model stays grounded in the retrieved evidence.

Generator Module
Uses a locally downloaded flan-t5-base model (via Hugging Face and LangChain) to generate well-structured answers using the prompt and retrieved content.

📊 Evaluation Strategy
Created a set of 20 real-world, representative questions.

Evaluated answers for:

Relevance

Grounding in context

Clarity

Used a table format with fields:

Question

Generated Answer

Retrieved Sources

Quality Score (1–5)

Comments

✅ Sample Results (Excerpt)
Question	Score	Comments
Do customers complain about unexpected fees?	4	Clear and supported
What are recurring issues in Buy Now, Pay Later services?	5	Highly accurate
Do users report identity theft?	3	Lacked enough evidence

💡 Recommendations
Area	Suggestion
🔁 Context Relevance	Add filtering or re-ranking to improve source precision
📦 Model Performance	Experiment with flan-t5-large or quantized Llama models
🧪 Evaluation	Introduce user feedback scoring and categorize question types
📐 Scaling	Add RAG output to dashboards or auto-tag complaints

💬 Task 4: Interactive Chat Interface (Streamlit)
🎯 Objective
To create a user-friendly interface for non-technical users (e.g., customer service teams, product managers) to interact with the RAG system.

🖥️ Features
🔎 Text input box for asking natural language questions

🤖 Model-generated answer display

📚 Source chunks shown under an expandable section

🧹 "Clear Chat History" button

💾 Local loading of the LLM (flan-t5-base) for fast and private inference

🚀 How to Run
Ensure you have the local model downloaded in:
models/flan-t5-base/

Activate the virtual environment and run:

streamlit run app.py
Interact with the assistant in the browser.


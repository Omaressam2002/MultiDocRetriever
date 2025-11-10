

# 🧠 MultiDocRetriever — Chat with YouTube Videos, PDFs & Texts!

🎥 Demo YouTube Video: [https://youtu.be/IaCaCOoBUXg]


### 🚀 An intelligent chatbot that understands your content — from YouTube videos to documents.

**MultiDocRetriever** is an AI-powered chatbot built with **LangChain**, **Streamlit**, and **Groq’s LLaMA-3.3-70B**.
It extracts knowledge from **YouTube captions**, **PDFs**, and **text files**, allowing you to chat naturally with your own content.

---

## 🌟 Features

✅ Upload **YouTube videos** by URL — captions are automatically extracted
✅ Upload **PDF** and **Text** documents — fully parsed and searchable
✅ Context-aware conversations powered by **Groq’s LLaMA-3.3-70B-Versatile**
✅ Uses **Pinecone** as a vector database for scalable semantic search
✅ **Reranking** with **MS MARCO MiniLM-L-6-v2** for improved relevance
✅ Built-in **LangGraph** agent router to dynamically decide retrieval paths
✅ Deployable anywhere using **Docker**

---

## 🧩 Tech Stack

| Component          | Description                        |
| ------------------ | ---------------------------------- |
| **Frontend**       | Streamlit (Python web UI)          |
| **LLM**            | Groq API — LLaMA-3.3-70B-Versatile |
| **Embeddings**     | e5-small                           |
| **Reranker**       | MS MARCO MiniLM-L-6-v2             |
| **Vector Storage** | Pinecone                           |
| **Framework**      | LangChain + LangGraph              |
| **Deployment**     | Docker                             |

---

## 🧠 System Architecture

The app is powered by a **LangGraph agent network**, orchestrating how queries are handled intelligently.

### 🔹 1. Router Node

The **Router Node** is an LLM-based decision maker.
It analyzes the user query and conversation context to decide whether **retrieval** is required.

* **If retrieval is *not* needed:**
  ➜ The query is sent directly to the **Answer Node**.

* **If retrieval *is* needed:**
  ➜ The query is sent to the **Retrieval Node** for document search.

---

### 🔹 2. Retrieval Node

This node retrieves **10–20 relevant documents** from the Pinecone index using **e5-small embeddings**.
It then passes them to the **Generation Node** for deeper reasoning.

---

### 🔹 3. Generation Node

The **Generation Node** performs:

1. **Reranking** with *MS MARCO MiniLM-L-6-v2*
2. Selects the **top 7 most relevant documents** (a tuned hyperparameter)
3. Combines:

   * The user query
   * Previous conversation context
   * The retrieved and reranked knowledge
   * → to generate a coherent and context-aware answer

---

### 🔹 4. Answer Node

If the Router decides no retrieval is necessary, the **Answer Node** generates a response purely based on the **LLM** and **conversation memory**.

---

## ⚙️ Workflow Summary

```text
User Query
   ↓
[Router Node] → decides whether retrieval is needed
   ├──→ [Answer Node] → direct answer from LLM
   └──→ [Retrieval Node] → fetches 10–20 docs
             ↓
        [Generation Node] → reranks top 7, merges context, produces final answer
```

---

## 🖼️ Architecture Diagram

The following figure illustrates the **LangGraph Agent Flow**:

![LangGraph Architecture](assets/Graph.png)

---

## 🐳 Running the Project with Docker

Build the image:

```bash
docker build -t multidocret .
```

Run the container:

```bash
docker run -p 8501:8501 multidocret
```

Access the app at:

```
http://localhost:8501
```

---

## 📚 How It Works in Action

1. **YouTube Mode:**

   * Paste a YouTube link → captions are extracted → AI ingests the content.
2. **Document Mode:**

   * Upload PDFs or text files → AI embeds and indexes them in Pinecone.




## 💡 Future Improvements

* Add **Data Analysis Agent** (using CSV and Excel files)

---




from typing import TypedDict, List, Tuple, Any
from utils import *  # add reload module
from modules import E5Embedder, MsMarcoCrossEncoder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain.schema import HumanMessage
import os


# allow each to have their prompt
# change the prompt for each of them to give context and also use an llm to construct a query or an instruction 
# rewrite query to extract only the

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = "multi-doc-index"

pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pc.Index(INDEX_NAME)§
reranker = MsMarcoCrossEncoder()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.6,
    api_key=os.getenv("GROQ_API_KEY")
)


class GraphState(TypedDict):
    question: str
    docs: List[str]
    answer: str
    error: str
    llm: ChatGroq
    route: str
    history: List[Tuple[str, str]]


# --------------------- ROUTER NODE ---------------------
def router_node(state: GraphState):
    try:
        # Ensure history exists
        state.setdefault("history", [])

        # Check if Pinecone index is empty
        if index.describe_index_stats()["total_vector_count"] == 0:
            state["route"] = "answer"
            return state

        router_llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            api_key=os.getenv("GROQ_API_KEY")
        )

        question = state["question"]
        prompt = (
            f"If I asked you this question: '{question}', and you have a knowledge "
            f"database for retrieval, respond with 'yes' if you would retrieve "
            f"information to answer, or 'no' if you would not."
        )

        response = router_llm.invoke([HumanMessage(content=prompt)])
        answer = response.content.strip()[:3].lower()

        if answer == "yes":
            state["route"] = "retrieve"
        elif answer == "no":
            state["route"] = "answer"
        else:
            state["route"] = "error"
            write_log(f"[ERROR IN ROUTER NODE] Invalid LLM output: {response.content}", level="error")

    except Exception as e:
        state["error"] = True
        write_log("[ERROR IN ROUTER NODE]:", level="error", exc=e)

    return state


# --------------------- ANSWER NODE ---------------------
def answer_node(state: GraphState):
    try:
        state.setdefault("history", [])
        if state["history"]:
            history_text = "\n".join(
                [f"User: {q}\nAssistant: {a}" for q, a in state["history"]]
            )
            prompt = (
                f"Answer the following question: {state['question']}\n\n"
                f"Previous conversation (if relevant):\n{history_text}"
            )
        else:
            prompt = f"Answer the following question: {state['question']}"

        response = llm.invoke([HumanMessage(content=prompt)])
        answer_text = response.content.strip()

        state["answer"] = response
        state["history"].append((state["question"], answer_text))

        write_log(f"ANSWERED WITHOUT RETRIEVAL: {prompt} → {answer_text[:100]}...")

    except Exception as e:
        state["error"] = True
        write_log("[ERROR IN ANSWER NODE]:", level="error", exc=e)

    return state


# --------------------- RETRIEVE NODE ---------------------
def retrieve_node(state: GraphState):
    try:
        query = "query: " + state["question"]

        vectorstore_pdf = PineconeVectorStore.from_existing_index(
            index_name=INDEX_NAME,
            embedding=E5Embedder(),
            namespace="pdf"
        )

        vectorstore_text = PineconeVectorStore.from_existing_index(
            index_name=INDEX_NAME,
            embedding=E5Embedder(),
            namespace="text"
        )

        pdf_results = vectorstore_pdf.similarity_search(query, k=10)
        text_results = vectorstore_text.similarity_search(query, k=10)

        all_results = pdf_results + text_results
        pairs = [(query, doc.page_content) for doc in all_results]

        scores = reranker.model.predict(pairs)
        scored_docs = sorted(zip(all_results, scores), key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, _ in scored_docs[:7]]

        state["docs"] = top_docs

    except Exception as e:
        state["error"] = True
        write_log("[ERROR IN RETRIEVE NODE]:", level="error", exc=e)

    return state


# --------------------- GENERATION NODE ---------------------
def generation_node(state: GraphState):
    try:
        state.setdefault("history", [])

        question = state["question"]
        docs = [x for x in state["docs"]]
        context = "\n\n".join([d.page_content for d in docs])

        if state["history"]:
            history_text = "\n".join(
                [f"User: {q}\nAssistant: {a}" for q, a in state["history"]]
            )
            prompt = f"""
Answer the following question using the context below. Mention the exact lines that helped you.

Question: {question}

Context:
{context}

Conversation so far:
{history_text}
"""
        else:
            prompt = f"""
Answer the following question using the context below. Mention the exact lines that helped you.

Question: {question}

Context:
{context}
"""

        response = llm.invoke([HumanMessage(content=prompt)])
        answer_text = response.content.strip()

        state["answer"] = response
        state["history"].append((state["question"], answer_text))

    except Exception as e:
        state["error"] = True
        write_log("[ERROR IN GENERATION NODE]:", level="error", exc=e)

    return state
def create_RAG_graph():
    builder = StateGraph(GraphState)

    builder.add_node("router", router_node)
    builder.add_node("answer", answer_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generation", generation_node)
    
    builder.set_entry_point("router")
    builder.add_conditional_edges("router",lambda state: state["route"],
        {
            "error": END,
            "retrieve" : "retrieve",
            "answer": "answer"
        }
        )
    builder.add_edge("answer", END)
    builder.add_conditional_edges("retrieve", lambda state: "error" if state.get("error") else "next",
        {
            "error": END,
            "next": "generation"
        }
    )

    builder.add_edge("generation", END)
    
    graph = builder.compile()
    return graph

class BaseAgent:
    def __init__(self):
        # Ensure index exists once
        if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
            pc.create_index(
                name=INDEX_NAME,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        
        self.graph = create_RAG_graph()

        # so that it doesnt split mid sentence
        self.splitter = RecursiveCharacterTextSplitter(
                            separators=["\n\n", ".", "!", "?", "\n", " "], 
                            chunk_size=1000,
                            chunk_overlap=50,
                            length_function=len
                        )
        self.embedder = E5Embedder()

    def get_response(self, query, history=None):
        if history is None:
            history = []
        result = self.graph.invoke({
            "question": query,
            "history": history
        })
        return result['answer'].content, result['history']

    def ingest_file(self, file_content, file_type, source_name):
        try:
            split_docs = self.splitter.split_documents(file_content)
            split_docs = prefix_passage_texts(split_docs, source_name)

            write_log(f"Started Embedding {len(split_docs)} chunks to {file_type.upper()} Chunks...")
            PineconeVectorStore.from_documents(
                split_docs,
                embedding=self.embedder,
                namespace=file_type,
                index_name=INDEX_NAME
            )
            write_log(f"Successfully Embedded {len(split_docs)} chunks")
        except Exception as e:
            write_log("[ERROR IN FILE INGESTING]:", level="error", exc=e)
            return False
        return True

    def delete_file(self, file_type, file_path):
        try:
            write_log(f"Removing File: {file_path}...")
            vectorstore = PineconeVectorStore(
                embedding=self.embedder,
                index_name=INDEX_NAME,
                namespace=file_type
            )
            vectorstore.delete(filter={"filename": file_path})
            write_log(f"Removed File: {file_path}")
        except Exception as e:
            write_log("[ERROR IN FILE DELETING]:", level="error", exc=e)
            return False
        return True
            

class DocAgent(BaseAgent):
    def __init__(self):
        super().__init__()

    def ingest_file(self, file_content, file_type, filename):
        return super().ingest_file(file_content, file_type ,filename)


class VidAgent(BaseAgent):
    def __init__(self):
        super().__init__(namespace="youtube")

    def ingest_file(self, file_content, file_type, file_name):
        return super().ingest_file(file_content, file_type , file_name)

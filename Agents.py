from typing import TypedDict, List, Tuple, Any
from utils import * 
from modules import E5Embedder, MsMarcoCrossEncoder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain.schema import HumanMessage
import os
from langchain_core.documents import Document

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = "multi-doc-index"

pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pc.Index(INDEX_NAME)
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
    agent_type: str 


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
        # provide history here too?
        prompt = (
            f"If I asked you this question: '{question}', and you have a knowledge "
            f"database, containing PDF documents and Youtube videos' captions, for retrieval, respond with 'yes' if you would needt to retrieve "
            f"information to answer, or 'no' if you would not."
            f"IMPORTANT : ANSWER ONLY WITH : 'YES' OR 'NO.' "
        )

        response = router_llm.invoke([HumanMessage(content=prompt)])
        answer = response.content.strip()[:3].lower()

        if answer == "yes":
            state["route"] = "retrieve"
        elif answer in ["no", "no."]:
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
        history_text = None
        if state["history"]:
            history_text = "\n".join(
                [f"User: {q}\nAssistant: {a}" for q, a in state["history"]]
            )
        prompt = f"""You are a helpful, friendly, and conversational AI assistant.
            
            Your role:
            - Answer questions accurately and clearly
            - Engage naturally with greetings and casual conversation
            - Appreciate humor and respond appropriately to jokes
            - Be concise but thorough when needed
            - Admit when you don't know something
            
            Context from previous conversation:
            {history_text if history_text else "This is the start of the conversation."}
            
            User: {state['question']}"""

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

        if state['agent_type'] == "Vid":
            vectorstore = PineconeVectorStore.from_existing_index(
            index_name=INDEX_NAME,
            embedding=E5Embedder(),
            namespace="youtube"
        )
            all_results = vectorstore.similarity_search(query, k=20)
        elif state['agent_type'] == "Doc" :
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

        if state["agent_type"] == "Vid" :
            if state["history"]:
                history_text = "\n".join(
                    [f"User: {q}\nAssistant: {a}" for q, a in state["history"]]
                )
                prompt = f"""You are a helpful AI assistant analyzing YouTube video content.
            
                        Your task: Answer the user's question based on the video captions provided below.
                        
                        Guidelines:
                        - The context comes from YouTube video captions/transcripts, which may contain:
                          * Speech-to-text errors or informal spoken language
                          * Filler words (um, uh, like, you know)
                          * Incomplete sentences or conversational patterns
                          * Time-stamped segments that may be out of perfect order
                        - Quote or paraphrase the relevant parts that support your answer
                        - If the captions don't contain the answer, clearly state this
                        - Mention approximately where in the video the information appears if possible
                        - Interpret the meaning even if the wording is imperfect
                        
                        Question: {question}
                        
                        Video Captions/Transcript:
                        {context}
                        
                        Previous Conversation:
                        {history_text}
                        
                        Answer:"""
            
            else:
                prompt = f"""You are a helpful AI assistant analyzing YouTube video content.
            
                        Your task: Answer the user's question based on the video captions provided below.
                        
                        Guidelines:
                        - The context comes from YouTube video captions/transcripts, which may contain:
                          * Speech-to-text errors or informal spoken language
                          * Filler words (um, uh, like, you know)
                          * Incomplete sentences or conversational patterns
                          * Time-stamped segments that may be out of perfect order
                        - Quote or paraphrase the relevant parts that support your answer
                        - If the captions don't contain the answer, clearly state this
                        - Mention approximately where in the video the information appears if possible
                        - Interpret the meaning even if the wording is imperfect
                        
                        Question: {question}
                        
                        Video Captions/Transcript:
                        {context}
                        
                        Answer:"""
        else : 
            if state["history"]:
                history_text = "\n".join(
                    [f"User: {q}\nAssistant: {a}" for q, a in state["history"]]
                )
                prompt = f"""You are a helpful AI assistant analyzing PDF document content.
                        
                        Your task: Answer the user's question based on the document excerpts provided below.
                        
                        Guidelines:
                        - The context comes from PDF documents, which may include:
                          * Formal written text, technical content, or structured information
                          * Tables, lists, headings, and organized sections
                          * Academic, business, or technical terminology
                          * References, citations, or footnotes
                        - Cite specific passages that support your answer
                        - Maintain the formality and precision of the source material
                        - If information spans multiple sections, synthesize them coherently
                        - If the documents don't contain the answer, clearly state this
                        - Preserve technical accuracy and specific terminology from the source
                        
                        Question: {question}
                        
                        Document Content:
                        {context}
                        
                        Previous Conversation:
                        {history_text}
                        
                        Answer:"""
            
            else:
                prompt = f"""You are a helpful AI assistant analyzing PDF document content.
                        
                        Your task: Answer the user's question based on the document excerpts provided below.
                        
                        Guidelines:
                        - The context comes from PDF documents, which may include:
                          * Formal written text, technical content, or structured information
                          * Tables, lists, headings, and organized sections
                          * Academic, business, or technical terminology
                          * References, citations, or footnotes
                        - Cite specific passages that support your answer
                        - Maintain the formality and precision of the source material
                        - If information spans multiple sections, synthesize them coherently
                        - If the documents don't contain the answer, clearly state this
                        - Preserve technical accuracy and specific terminology from the source
                        
                        Question: {question}
                        
                        Document Content:
                        {context}
                        
                        Answer:"""

        
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
    def __init__(self, agent_type='Doc'):
        # Ensure index exists once
        if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
            pc.create_index(
                name=INDEX_NAME,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        
        self.graph = create_RAG_graph()
        self.agent_type = agent_type

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
            "history": history,
            "agent_type": self.agent_type
        })
        return result['answer'].content, result['history']

    def ingest_file(self, file_content, file_type, source_name):
        try:

            if self.agent_type == "Vid":
                file_content = [Document(page_content=file_content)]
            split_docs = self.splitter.split_documents(file_content)
            split_docs = prefix_passage_texts(split_docs, source_name)
            write_log(f"Started Embedding {len(split_docs)} chunks to {file_type.upper()} Chunks...")
            PineconeVectorStore.from_documents(
                split_docs,
                embedding=self.embedder,
                namespace=file_type,
                index_name=INDEX_NAME
            )
            print(f"Started Embedding {len(split_docs)} chunks to {file_type.upper()} Chunks...")
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
        super().__init__(agent_type = 'Doc')

    def ingest_file(self, file_content, file_type, filename):
        return super().ingest_file(file_content, file_type ,filename)


class VidAgent(BaseAgent):
    def __init__(self):
        super().__init__(agent_type = 'Vid')

    def ingest_file(self, file_content, file_type, file_name):
        return super().ingest_file(file_content, file_type , file_name)

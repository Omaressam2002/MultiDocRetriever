from Agents import DocAgent
import streamlit as st
from streamlit_chat import message
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from utils import *
import tempfile


# streaming letters and better handling at wait time
st.set_page_config(page_title="ChatGPT-like Chatbot", page_icon="📄", layout="centered")
st.title("📄 PDF and Text Chatbot")
st.caption("A minimal ChatGPT-style chatbot built with Streamlit.")

# --- Initialize session state ---
if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = []  # names of currently ingested files
    
if 'history' not in st.session_state:
    st.session_state['history'] = []

if "messages" not in st.session_state:
    st.session_state["messages"] = [] 
    
if "agent" not in st.session_state:
    st.session_state["agent"] = DocAgent()


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

Agent = st.session_state["agent"]


# --- Upload UI ---
uploaded_files = st.sidebar.file_uploader("Upload", type=["pdf", "txt"], accept_multiple_files=True)

# --- Detect added or removed files ---
current_filenames = [f.name for f in uploaded_files] if uploaded_files else []
previous_filenames = st.session_state["uploaded_files"]

added_files = [f for f in current_filenames if f not in previous_filenames]
removed_files = [f for f in previous_filenames if f not in current_filenames]

# --- Handle newly added files ---
for uploaded_file in uploaded_files or []:
    if uploaded_file.name in added_files:
        file_type = uploaded_file.name.split('.')[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        if file_type == "pdf":
            loader = PyPDFLoader(tmp_path)
        else:
            loader = TextLoader(tmp_path)
        Agent.ingest_file(loader.load(), file_type, uploaded_file.name)
        write_log(f"File Ingested: {uploaded_file.name}")

# --- Handle removed files ---
for removed_file in removed_files:
    file_type = removed_file.split('.')[-1].lower()
    Agent.delete_file(file_type, removed_file)
    write_log(f"File Deleted: {removed_file}")

# --- Update state ---
st.session_state["uploaded_files"] = current_filenames
      
def conversational_chat(query):
    result, history = Agent.get_response(query, st.session_state['history'])
    st.session_state['history'] = history 
    return result



if user_input := st.chat_input("Type your message here..."):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(message["content"])

    response = conversational_chat(user_input)

    # Append assistant message
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)


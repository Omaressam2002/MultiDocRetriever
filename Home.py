from Agents import DocAgent
import streamlit as st
from streamlit_chat import message
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from utils import *
import tempfile
import importlib
import sys
 

# then 3awzeen nezaker RAN we micro ai interview we mcp agentic comp we freelancing we mentorees we pdf picture agents we time series we ecg



def reload_all_modules():
    """Reload all local project modules."""
    modules_to_reload = [
        "Agents.DocAgent",
        "utils.*",
    ]
    for module_name in modules_to_reload:
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])

reload_all_modules()

# streaming letters and better handling at wait time
st.set_page_config(page_title="ChatGPT-like Chatbot", page_icon="📄", layout="centered")
st.title("📄 PDF and Text Chatbot")
st.caption("A minimal ChatGPT-style chatbot built with Streamlit.")

# --- Initialize session state ---
if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = []  # names of currently ingested files
    
if 'Doc_history' not in st.session_state:
    st.session_state['Doc_history'] = []

if "Doc_messages" not in st.session_state:
    st.session_state["Doc_messages"] = [{"role": "assistant" , "content" : "Hello 👋, How Can I Help You Today?"}] 
    
if "Doc_Agent" not in st.session_state:
    st.session_state["Doc_Agent"] = DocAgent()


Agent = st.session_state["Doc_Agent"] 


# --- Upload UI ---
if st.sidebar.button("🔄 Reset Chat"):
    st.session_state.clear()
    st.rerun()
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
    result, history = Agent.get_response(query, st.session_state['Doc_history'])
    st.session_state['Doc_history'] = history 
    return result

for message in st.session_state['Doc_messages']:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Type your message here..."):
    # Append user message
    st.session_state['Doc_messages'].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    response = conversational_chat(user_input)

    # Append assistant message
    st.session_state['Doc_messages'].append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)


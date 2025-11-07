from Agent import PDFAgent
import streamlit as st
from streamlit_chat import message
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from utils import *
import tempfile

# streaming letters and better handling at wait time

# --- Initialize session state ---
if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = []  # names of currently ingested files
    
if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = []
    # "Hello ! Ask me anything about these files 🤗"

if 'past' not in st.session_state:
    st.session_state['past'] = []
    #"Hey ! 👋"
    
if "agent" not in st.session_state:
    st.session_state["agent"] = PDFAgent()

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

    
#container for the chat history
response_container = st.container()
#container for the user's text input
container = st.container()


with container:
    with st.form(key='my_form', clear_on_submit=True):
        
        user_input = st.text_input("Query:", placeholder="Ask me anything about the PDF (:", key='input')
        submit_button = st.form_submit_button(label='Send')
        
    if submit_button and user_input:
        output = conversational_chat(user_input)
        
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
            message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
            

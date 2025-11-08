from utils import get_transcript
import streamlit as st
from streamlit_chat import message
import time
from Agent import VidAgent
from utils import *


# seperate the two chatbots



# ---------- Streamlit Page Config ----------
st.set_page_config(page_title="YouTube Chatbot", page_icon="📹", layout="centered")
st.title("📹 YouTube Chatbot")
st.caption('upload your link and ask your questions')

# ---------- Initialize Session State ----------
if 'history' not in st.session_state:
    st.session_state['history'] = []

if "messages" not in st.session_state:
    st.session_state["messages"] = [] 

if "videos" not in st.session_state:
    st.session_state["videos"] = []  # {"url": ..., "title": ..., "transcript": ...}

if "agent" not in st.session_state:
    st.session_state["agent"] = VidAgent()

Agent = st.session_state["agent"]

# ---------- Sidebar UI for YouTube Upload ----------
st.sidebar.header("🎬 Video Loader")
with st.sidebar.form(key="youtube_form",clear_on_submit=True):
    video_url = st.text_input("Enter YouTube Video URL:")  # clear
    submit = st.form_submit_button("Add Video")


# ---------- Chat Message Display ----------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if submit:
    if video_url:
        # Check if URL already exists
        if any(v["url"] == video_url for v in st.session_state["videos"]):
            st.sidebar.warning("⚠️ This video is already added.")
        else:
            try:
                res = get_transcript(video_url)
                st.session_state["videos"].append({
                    "url": video_url,
                    "title": res["title"],
                    "transcript": res["text"]
                })
                st.sidebar.success(f"✅ Added: {res['title']}") # i want this to be for 5 seconds only
                file_content = res["title"] + ": "+ res["text"]
                Agent.ingest_file(file_content, "youtube", res["title"])
                write_log(f"File Ingested: {res['title']}")
            except Exception as e:
                st.sidebar.error(f"Failed to load transcript: {e}")
                write_log("[ERROR IN FILE INGESTING] :", level="error", exc=e)
    else:
        st.sidebar.warning("Please enter a valid YouTube URL.")

# ---------- Display List of Loaded Videos ----------
st.sidebar.subheader("📄 Loaded Videos")
if st.session_state["videos"]:
    for i, video in enumerate(st.session_state["videos"]):
        col1, col2 = st.sidebar.columns([4, 1])
        with col1:
            st.markdown(f"**🎥 {video['title']}**")
        with col2:
            remove = st.button("✖", key=f"remove_{i}", help="Remove video")
            if remove:
                removed_title = video["title"]
                st.session_state["videos"].pop(i)

                info_placeholder = st.sidebar.empty()
                st.sidebar.error(f"❌ Removing: {video['title']}")
                Agent.delete_file("youtube", removed_title)
                write_log(f"File Deleted: {removed_title}")
                info_placeholder.empty()

                st.rerun()
else:
    st.sidebar.info("No videos added yet. Add a YouTube link above!")

def conversational_chat(query):
    result, history = Agent.get_response(query, st.session_state['history'])
    st.session_state['history'] = history 
    return result

if user_input := st.chat_input("Type your message here..."):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    response = conversational_chat(user_input)

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
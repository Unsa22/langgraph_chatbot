import streamlit as st
from chatbot import workflow,retrieve
from langchain_core.messages import HumanMessage
import uuid

#_______________________________________________utility function__________________________________________
def generate_id():
    id= uuid.uuid4()
    return id

def reset_chat():
    thread_id= generate_id()
    st.session_state["thread_id"]= thread_id
    add_threads(st.session_state["thread_id"])
    st.session_state["message_history"]=[]

def add_threads(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

def load_conversation(thread_id):
    return workflow.get_state(config={"configurable":{"thread_id":thread_id}}).values["messages"]

   

#_____________________________Session setup______________________________________________________________
if "message_history" not in st.session_state:
    st.session_state["message_history"]=[]

if "thread_id" not in st.session_state:
    st.session_state["thread_id"]=generate_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"]=retrieve()

add_threads(st.session_state["thread_id"])

#_____________________________Side bar UI________________________________________________________________
st.sidebar.title("chatbot")

if st.sidebar.button("new chat"):
    reset_chat()

st.sidebar.header("conversation")

for thread_id in st.session_state["chat_threads"][::-1]:
    if st.sidebar.button(str(thread_id)):
        st.session_state["thread_id"]= thread_id
        messages= load_conversation(thread_id)
        temp_msg= []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role= "user"
            else: 
                role= "assistant"
            temp_msg.append({"role":role, "content":msg.content})      
        st.session_state["message_history"]= temp_msg  



#_____________________________main UI____________________________________________________________________

for messages in st.session_state["message_history"]:
    with st.chat_message(messages["role"]):
        st.text(messages["content"])

user_input= st.chat_input("Type: ")
CONFIG={"configurable":{"thread_id":st.session_state["thread_id"]},
        "metadata":{"thread_id":st.session_state["thread_id"]},
        "run_name":"chat_turn"}
if user_input:
    st.session_state["message_history"].append({"role":"user","content":user_input})
    with st.chat_message("user"):
        st.text_input(user_input)
    with st.chat_message('assistant'):
        ai_msg= st.write_stream(
            message_chunk.content for message_chunk, metadata in workflow.stream(
                {
                    "messages": HumanMessage(content=user_input)
                    },
                    config= CONFIG,
                    stream_mode= "messages"
                    ))
    st.session_state["message_history"].append({"role":"assistant","content":ai_msg})




from utils import chain, index, chat_history
import streamlit as st
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
import os

from streamlit_chat import message


st.subheader("Chatbot Based On Personalized Data using LLMs")


# Open the file in read mode ('r')
with open('data/data.txt', 'r') as file:
    # Read the contents of the file into the 'data' variable
    jotaro_description = file.read()

# uploaded_file = st.file_uploader("Choose a text file", type="txt")

# if uploaded_file is not None:
#     text = uploaded_file.read().decode('utf-8')
#     st.write(text)

st.markdown("""
    <div style="color:#202124;padding:10px;border:2px solid #ff6b55; border-radius:5%; background-color:beige;">
            <h3 style="color:#ff6b55">Personalized Data:</h3>
            """+jotaro_description+"""</div><br>""", unsafe_allow_html=True)

# st.write(jotaro_description)


agree = st.checkbox('Use The whole model !')

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []


# container for chat history
response_container = st.container()
# container for text box
textcontainer = st.container()
st.session_state["query"] = ""

with textcontainer:
    query = st.text_input(
        "Query: ", value=st.session_state["query"], key="input")
    if query:
        with st.spinner("typing..."):
            if agree:
                response = chain(
                    {"question": query, "chat_history": chat_history})['answer']
                chat_history.append((query, response))
            else:
                response = index.query(query)
            st.session_state["query"] = ""
        st.session_state.requests.append(query)
        st.session_state.responses.append(response)

with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i], key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i],
                        is_user=True, key=str(i) + '_user')

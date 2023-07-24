
from langchain.vectorstores import Chroma
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import sys
import os

import streamlit as st

print("UTIL LOADED!")

os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]

# Enable to save to disk & reuse the model (for repeated queries on the same data)
PERSIST = False
chat_history = []
query = None

if len(sys.argv) > 1:
    query = sys.argv[1]

if PERSIST and os.path.exists("persist"):
    print("Reusing index...\n")
    vectorstore = Chroma(persist_directory="persist",
                         embedding_function=OpenAIEmbeddings())
    index = VectorStoreIndexWrapper(vectorstore=vectorstore)
else:
    # Use this line if you only need data.txt
    #loader = TextLoader("data/data.txt")
    loader = DirectoryLoader("data/", glob="*.txt")
    if PERSIST:
        index = VectorstoreIndexCreator(
            vectorstore_kwargs={"persist_directory": "persist"}).from_loaders([loader])
    else:
        index = VectorstoreIndexCreator().from_loaders([loader])

chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
)

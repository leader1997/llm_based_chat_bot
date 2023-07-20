import logging

import os
import sys

from langchain.document_loaders import TextLoader,DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

from apikey import  OPENAI_API_KEY

query='who is obama?'

os.environ['OPENAI_API_KEY']= OPENAI_API_KEY

loader=DirectoryLoader(".",glob="data/*.txt")
index=VectorstoreIndexCreator().from_loaders([loader])

print(index.query(query,llm=ChatOpenAI()))

import os
import logging, datetime

from pymilvus import connections
from milvus import default_server
from langchain_openai import OpenAIEmbeddings
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# for PDF Download 
import tempfile
from langchain.document_loaders import PyPDFLoader

now = datetime.datetime.now()
formatted_datetime = now.strftime("%d-%m-%Y_%H%M")
logging.basicConfig(filename=f'log/emb_{formatted_datetime}.log', 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

OPENAI_API_KEY=     os.getenv("OPENAI_API_KEY")

@st.cache_data
def connect_to_milvus():
    print('connecting to milvus lite...')
    port_no = default_server.listen_port
    connections.connect(host= 'localhost', port=str(port_no))
    print("Milvus connected")


@st.cache_resource
def connect_openai_llm():
    client = OpenAI()
    return client

@st.cache_resource
def connect_openai_embedding():
    openai_embedding = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=768
)
    return openai_embedding

import pandas as pd
import string
import random
import os
import logging, datetime

from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema,DataType
from milvus import default_server
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_ibm import WatsonxEmbeddings
from langchain_openai import OpenAIEmbeddings
import streamlit as st
from dotenv import load_dotenv
from ibm_watsonx_ai.client import APIClient
from ibm_watsonx_ai.foundation_models import Model
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


@st.cache_data
def read_pdf(uploaded_files):
    for uploaded_file in uploaded_files:
      bytes_data = uploaded_file.read()
      with tempfile.NamedTemporaryFile(mode='wb', delete=False) as temp_file:
          temp_file.write(bytes_data)
          filepath = temp_file.name
          with st.spinner('Waiting for the file to upload'):
            loader = PyPDFLoader(filepath)
            data = loader.load()
            docs = format_pdf_reader(data)
            return docs

@st.cache_data
def initiate_username():
    characters = string.ascii_letters + string.digits + '_'
    username = ''.join(random.choice(characters) for _ in range(random.randint(5, 32)))
    print('initiate username....')
    return 'a'+ username

#------create milvus database
def create_milvus_db(collection_name):
    item_id    = FieldSchema( name="id",         dtype=DataType.INT64,    is_primary=True, auto_id=True )
    text       = FieldSchema( name="text",       dtype=DataType.VARCHAR,  max_length= 50000             )
    embeddings = FieldSchema( name="embeddings", dtype=DataType.FLOAT_VECTOR,    dim=768                )
    schema     = CollectionSchema( fields=[item_id, text, embeddings], description="Inserted policy from user", enable_dynamic_field=True )
    collection = Collection( name=collection_name, schema=schema, using='default' )
    return collection


#----------split data using Langchain textspliter
def import_text_splitter(chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
        )
    return text_splitter


def split_text_with_overlap(text, chunk_size, overlap_size):
    chunks = []
    start_index = 0

    while start_index < len(text):
        end_index = start_index + chunk_size
        chunk = text[start_index:end_index]
        chunks.append(chunk)
        start_index += (chunk_size - overlap_size)

    return chunks


def embedding_data(chunks, collection_name, model_emb):
    print('embedding...')
    vector  = model_emb.embed_documents(chunks) # embedding chunks of text
    logging.info('no. of chunks : ', len(chunks))
    logging.info('storing data in Milvus vector database...')
    logging.info([chunks,vector])
    collection = create_milvus_db(collection_name)
    collection.insert([chunks,vector])
    collection.create_index(field_name="embeddings",
                            index_params={"metric_type":"IP","index_type":"IVF_FLAT","params":{"nlist":16384}})
    return collection


def find_answer(question, collection, model_emb):
    embedded_vector  = model_emb.embed_query(question)    # embedding question
    print('embedding question...')
    collection.load()           # query data from collection
    hits = collection.search(data=[embedded_vector], anns_field="embeddings", param={"metric":"IP","offset":0},
                    output_fields=["text"], limit=15)
    return hits


#------upload pdf
def format_pdf_reader(raw_data):
    # format content from pdf into text
    pdf_text = ""
    for data in raw_data:
        pdf_text+=data.page_content+"\n"
    return pdf_text

# #--------generate promt reday to prompt in model

# def generate_prompt_en(question, context, model_type="llama-2"):
#     output = f'''<|begin_of_text|><|start_header_id|>**system<|end_header_id|>`
# You are a helpful, respectful Thai assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.

# You will receive HR Policy on user queries HR POLICY DETAILS, and QUESTION from user below. Answer the question in Thai.

# HR POLICY DETAILS:
# {context}
# QUESTION: {question}

# Answer the QUESTION use details about HR Policy from HR POLICY DETAILS, explain your reasonings if the question is not related to REFERENCE please Answer
# “I don’t know the answer, it is not part of the provided HR Policy”
# <|eot_id|><|start_header_id|>user<|end_header_id|>

# hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>

# hello, I'm your HR Policy Assistance, please type your question<|eot_id|><|start_header_id|>user<|end_header_id|>

# {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
# '''
#     return output   

def generate_answer(openai_client, prompt):
    completion = openai_client.chat.completions.create(
        model="gpt-4o",
        messages= prompt
    )
    return completion.choices[0].message.content




def create_hits_dataframe(hits, num_hits=10):
    if len(hits[0]) < 10:
        num_hits = len(hits[0])
    dict_display = {
        f'chunk{i}': [hits[0][i].text]
        for i in range(num_hits)
    }
    df = pd.DataFrame(dict_display).T
    df.columns = ['Reference from document']
    return df

def display_hits_dataframe(hits, num_hits=10, width=1000):
    df_dis = create_hits_dataframe(hits, num_hits)
    st.dataframe(df_dis, width=width)
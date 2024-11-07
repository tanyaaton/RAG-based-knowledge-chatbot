import pandas as pd
import string
import random
import logging, datetime
import os

from pymilvus import utility, Collection, CollectionSchema, FieldSchema,DataType
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st

# for PDF Download 
import tempfile
from langchain.document_loaders import PyPDFLoader

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

def initiate_username():
    characters = string.ascii_letters + string.digits + '_'
    username = ''.join(random.choice(characters) for _ in range(random.randint(5, 32)))
    print('initiate username....')
    return 'a'+ username

def create_milvus_db(collection_name, file_name):
    item_id    = FieldSchema( name="id",         dtype=DataType.INT64,    is_primary=True, auto_id=True )
    text       = FieldSchema( name="text",       dtype=DataType.VARCHAR,  max_length= 50000             )
    embeddings = FieldSchema( name="embeddings", dtype=DataType.FLOAT_VECTOR,    dim=768                )
    schema     = CollectionSchema( fields=[item_id, text, embeddings], description=file_name, enable_dynamic_field=True )
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


def embedding_data(chunks, collection_name, model_emb, file_name):
    print('embedding...')
    vector  = model_emb.embed_documents(chunks) # embedding chunks of text
    logging.info('storing data in Milvus vector database...')
    logging.info([chunks,vector])
    collection = create_milvus_db(collection_name, file_name)
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
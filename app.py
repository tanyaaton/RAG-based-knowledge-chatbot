import logging
import datetime
import os

#for UI
import streamlit as st
from langchain.callbacks import StdOutCallbackHandler
from PIL import Image

# for Milvus 
from pymilvus import utility, Collection

# for function
from connection import (connect_to_milvus, connect_openai_llm, connect_openai_embedding)
from function import (initiate_username, read_pdf, create_milvus_db, split_text_with_overlap,
            embedding_data, find_answer,generate_answer, display_hits_dataframe)
from prompt import generate_pdf_prompt


#---------- settings ----------- #
model_id_llm='gpt-4o'
model_id_emb="text-embedding-3-large"

# Most GENAI logs are at Debug level.
now = datetime.datetime.now()
formatted_datetime = now.strftime("%d-%m-%Y_%H%M")
logging.basicConfig(filename=f'log/app_{formatted_datetime}.log', 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

st.set_page_config(
    page_title="Retrieval Augmented Generation",
    page_icon="🍰",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.header("Retrieval Augmented Generation 💬")

connect_to_milvus()
# model_llm = connect_watsonx_llm(model_id_llm)
model_llm = connect_openai_llm()
model_emb = connect_openai_embedding()

handler = StdOutCallbackHandler()

# Sidebar contents
# Sidebar contents
with st.sidebar:
    st.title("🌷Welcome")
    st.markdown('''
    This is your RAG App!
    ### Model Information
    - Embedding Model: `text-embedding-3-large`
    - LLM Model: `gpt-4o`
                
    ### Technology Stack:
    - **Advanced Search**:
        - 🐟 [Milvus Database](https://milvus.io/docs)
        - 🦜 [Langchain-openai Embedding](https://python.langchain.com/docs/integrations/text_embedding/openai/)
    - **LLM Integration**:
        - 🍀 [OpenAI LLM](https://openai.com/index/openai-api/)
    ''')


#===========================================================================================

username = initiate_username()
logging.info(username)

if uploaded_files := st.file_uploader("please drop your PDF file", accept_multiple_files=True):
    print(utility.list_collections())
    print('======',username,'======')
    if (username in utility.list_collections()):
        print('----- collection already exist')
        collection = Collection(username)
    else:
        text = read_pdf(uploaded_files)
        chunks = split_text_with_overlap(text, 1000, 300)
        logging.info(chunks)
        print('----- create new collection')
        collection = create_milvus_db(username)
        print(collection)
        collection = embedding_data(chunks, username, model_emb)
else:
    utility.drop_collection(f'{username}')
    print('dropped collection')

if uploaded_files :
    print('ready for input...')
    if user_question := st.text_input(
        "Ask any question:"
    ): 
        print('processing...')
        logging.info(user_question)
        hits = find_answer(user_question, collection, model_emb)
        prompt = generate_pdf_prompt(user_question, [hits[0][i].text for i in range(4)])
        logging.info(prompt)
        response = generate_answer(model_llm,prompt)
        st.text_area(label="Model Response", value=response, height=300)
        display_hits_dataframe(hits)

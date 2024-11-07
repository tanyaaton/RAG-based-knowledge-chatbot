import logging
import datetime
import os
import tempfile

import streamlit as st

from connection import (connect_to_milvus, connect_openai_llm, connect_openai_embedding)
from function import (initiate_username, read_pdf, create_milvus_db, split_text_with_overlap,
            embedding_data, find_answer,generate_answer, display_hits_dataframe)
from prompt import generate_pdf_prompt
from document import process_with_loader, adjust_chunk_size, split_using_markdown

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
model_llm = connect_openai_llm()
model_emb = connect_openai_embedding()


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


# Initialize session state for tracking files with unique usernames
if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = {}

# File upload section
# st.subheader("🗂️ Upload a New PDF File")
uploaded_file = st.file_uploader("Drop your PDF file here", accept_multiple_files=False)

if uploaded_file:
    file_name = uploaded_file.name
    username = initiate_username()  # Generate a unique username for each file
    print(f'Generated Username: {username}, File Name: {file_name}')

    # Check if the file has already been processed
    if file_name in st.session_state["uploaded_files"]:
        collection = st.session_state["uploaded_files"][file_name]["collection"]
        st.info(f"The file '{file_name}' has already been uploaded.")
    else:
        # Process and embed the document
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        with st.spinner("Analyzing document..."):
            docs = process_with_loader(temp_file_path)
            splits = split_using_markdown(docs)
            chunks = adjust_chunk_size(splits)

            # Create a new collection in Milvus
            # collection = create_milvus_db(username, file_name)
            collection = embedding_data(chunks, username, model_emb, file_name)
            print('----- New collection created')

        # Store the unique username and collection in session state
        st.session_state["uploaded_files"][file_name] = {
            "username": username,
            "collection": collection
        }
    # st.success(f"File '{file_name}' has been processed and stored.")

# File selection section
if st.session_state["uploaded_files"]:
    selected_file = st.selectbox("Or select a previous file to use:", options=list(st.session_state["uploaded_files"].keys()))

    if selected_file:
        selected_data = st.session_state["uploaded_files"][selected_file]
        collection = selected_data["collection"]
        print('===',collection, selected_data)
        # st.info(f"You selected the file: {selected_file}")

        # Question answering section for the selected file
        st.subheader("📬 Ask any question based on the document")
        if user_question := st.text_input("Question"):
            hits = find_answer(user_question, collection, model_emb)
            prompt = generate_pdf_prompt(user_question, [hits[0][i].text for i in range(4)])
            response = generate_answer(model_llm, prompt)
            st.text_area(label="Model Response", value=response, height=300)
            display_hits_dataframe(hits)
else:
    st.warning("No previously uploaded files available.")
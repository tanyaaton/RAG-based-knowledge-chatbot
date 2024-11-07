# 💬 RAG based knowledge chatbot ✨

This Retrieval-Augmented Generation (RAG) application utilizes **Azure Document Intelligence** for multi-format document processing, **OpenAI** for LLM and Embedding, **Milvus Lite** for vector database for storing and retrieving document embeddings, and **Streamlit** for the user interface.

### Libraries to Install
```bash
pip install streamlit milvus langchain openai python-dotenv
```

## Project Setup

### 1. Clone the Repository

```bash
git clone https://github.com/tanyaaton/RAG-based-knowledge-chatbot
cd RAG-based-knowledge-chatbot
```

### 2. Install Required Libraries

Install all required libraries using requirements.txt file:

```bash
pip install -r requirements.txt
```
additionally also install the following libraries
```bash
pip install -U pymilvus
pip install -U langchain-community
```

This will ensure all necessary packages are available for running the RAG application, including document processing, vector storage, and LLM integration.

### 2. Set Up Environment Variables

Create a `.env` file in the root directory to store sensitive information such as API keys and endpoints:
- OpenAI sign up [here](https://platform.openai.com/signup)
- Azure Document Intelligence sign up [here](https://azure.microsoft.com/en-us/products/ai-services/ai-document-intelligence)
```plaintext

# OpenAI API Key
OPENAI_API_KEY=<your-openai-api-key>

# Azure Document Intelligence
AZURE_DOC_ENDPOINT=<your-azure-doc-intelligence-endpoint>
AZURE_DOC_KEY=<your-azure-doc-intelligence-key>
```

### 3. Configure Milvus Lite

Milvus Lite can be installed and run locally to manage vector embeddings for the document chunks. Follow the [Milvus Lite installation guide](https://milvus.io/docs/install_standalone-docker.md) for detailed instructions.

After installation, configure the Milvus connection in the `connection.py` file to match your setup.

### 4. Running the Application
First run the following command to activate Milvus Lite:
```bash
milvus-server --proxy-port 19530
```

Open new terminal. Tostart the Streamlit application, run:
```bash
streamlit run app.py
```

The app interface will load in your default web browser.

## Key Functionalities

- **Document Upload**: Users can upload a new document or select previously uploaded documents from the session.
- **Document Processing**: Documents are processed using Azure Document Intelligence, and text is split into structured chunks.
- **Embedding and Storage**: The content chunks are embedded using OpenAI’s embedding model and stored in Milvus.
- **Question Answering**: Users can ask questions, and the app will retrieve relevant document chunks to generate an answer.


## Additional Notes

Ensure your API keys and endpoints are active and have sufficient usage limits to handle document processing and question-answering tasks.

For any troubleshooting or additional setup information, consult the documentation provided for Streamlit, Milvus, Azure, and OpenAI APIs.

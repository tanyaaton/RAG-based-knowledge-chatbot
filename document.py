import os
from dotenv import load_dotenv

load_dotenv()
doc_intelligence_endpoint = os.getenv("AZURE_DOC_ENDPOINT")
doc_intelligence_key = os.getenv("AZURE_DOC_KEY")

from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter


def split_using_markdown(docs):
# Split the document into chunks base on markdown headers.
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    docs_string = docs[0].page_content
    splits = text_splitter.split_text(docs_string)

    print("Length of splits: " + str(len(splits)))
    return splits


def process_with_loader(file_path):
    loader = AzureAIDocumentIntelligenceLoader(
        file_path=file_path,
        api_key=doc_intelligence_key,
        api_endpoint=doc_intelligence_endpoint,
        api_model="prebuilt-layout"  # or another model, based on your needs
    )
    docs = loader.load()  # Load the document using AzureAIDocumentIntelligenceLoader
    return docs

def adjust_chunk_size(splits):
    i=0
    # a=0
    chunk_list = []
    while i < len(splits):
        # print(f'-------- chunk{a} ---------')
        metadata = splits[i].metadata
        content = splits[i].page_content
        while len(content) <500 and i<len(splits)-1:
            if len(content+splits[i+1].page_content)>1000:  
                break
            else: 
                i +=1
                content = content+splits[i].page_content
        # print(len(content))
        if '<table>' in content:
            table = '{Table}\n'
        else: table =''
        if '<figure>' in content:
            figure = '{Figure}\n' 
        else: figure =''
        chunk = str(metadata)+'\n'+table+figure+content
        print(chunk)
        i+=1
        # a+=1
        chunk_list.append(chunk)
    return chunk_list
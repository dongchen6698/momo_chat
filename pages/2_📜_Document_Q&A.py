import streamlit as st
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

import os
from dotenv import load_dotenv
load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
OPENAI_API_VERSION = os.getenv('OPENAI_API_VERSION')
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
API_DEPLOYMENT_NAME = os.getenv('API_DEPLOYMENT_NAME')

# 初始化ChatModel
if "chat_model" not in st.session_state:
    chat_model = AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        openai_api_version=OPENAI_API_VERSION,
        openai_api_key=AZURE_OPENAI_API_KEY,
        deployment_name=API_DEPLOYMENT_NAME,
        temperature=0,
        streaming=True,
    )


if "embedding_model" not in st.session_state:
    embedding_model = AzureOpenAIEmbeddings(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        openai_api_key=AZURE_OPENAI_API_KEY,
        azure_deployment="ada2",
        openai_api_version=OPENAI_API_VERSION,
    )
embedding_model = embedding_model


# Page title
st.set_page_config(page_title="📜_Document_Q&A")
st.title("📜_Document_Q&A")

with st.sidebar:
    st.title("DocumentQA Introduction")
    st.markdown(
        """
        Upload a file and ask question!
        """
    )

# File upload
uploaded_file = st.file_uploader("Upload file.", type="txt")
if "retriever" not in st.session_state and uploaded_file is not None:
    # documents = [PDFPlumberLoader(uploaded_file).load()]
    documents = [uploaded_file.read().decode()]
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.create_documents(documents)
    # Create a vectorstore from documents
    db = Chroma.from_documents(texts, embedding_model)
    # Create retriever interface
    retriever = db.as_retriever()
    st.session_state.retrieval_qa_chain = RetrievalQA.from_chain_type(llm=chat_model, chain_type="stuff", retriever=retriever)

# Query text
query_text = st.text_input("Enter your question:", disabled=not uploaded_file)

# Form input and query
result = []
with st.form("myform", clear_on_submit=True):
    if "retrieval_qa_chain" in st.session_state:
        response = st.session_state.retrieval_qa_chain.run(query_text)
        result.append(response)

if len(result):
    st.info(response)

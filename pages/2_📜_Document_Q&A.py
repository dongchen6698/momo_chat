import os
import base64
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.vectorstores import Chroma
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from utils import login_auth

st.set_page_config(page_title="MultiDocument Chat", page_icon="🤗")

load_dotenv()
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
API_DEPLOYMENT_NAME = os.getenv("API_DEPLOYMENT_NAME")

def run_page():
    if "chat_model" not in st.session_state:
        st.session_state.chat_model = AzureChatOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            openai_api_version=OPENAI_API_VERSION,
            openai_api_key=AZURE_OPENAI_API_KEY,
            deployment_name=API_DEPLOYMENT_NAME,
            temperature=0,
            streaming=True,
        )


    if "embedding_model" not in st.session_state:
        st.session_state.embedding_model = AzureOpenAIEmbeddings(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            openai_api_key=AZURE_OPENAI_API_KEY,
            azure_deployment="ada2",
            openai_api_version=OPENAI_API_VERSION,
        )

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None


    def get_documents_texts(documents):
        texts = ""
        for doc in documents:
            for page in PdfReader(doc).pages:
                texts += page.extract_text()
        return texts


    def get_text_chunks(doc_texts):
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        text_chunks = text_splitter.split_text(doc_texts)
        return text_chunks


    def get_vectorstore(chunks):
        vectorstore = Chroma.from_texts(chunks, embedding=st.session_state.embedding_model)
        return vectorstore


    def get_conversation_chain(vectorstore):
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=st.session_state.chat_model,
            retriever=vectorstore.as_retriever(),
            memory=memory,
        )
        return conversation_chain


    def handle_userinput(user_question):
        if st.session_state.chat_history is not None:
            for message in st.session_state.chat_history:
                if message.type not in ("system"):
                    with st.chat_message(message.type):
                        st.markdown(message.content)

        with st.chat_message("user"):
            st.markdown(user_question)
        with st.chat_message("assistant"):
            response = st.session_state.conversation({"question": user_question})
            st.markdown(response["answer"])


        st.session_state.chat_history = response["chat_history"]


    st.header("Chat with multiple PDFs :books:")
    user_question = st.chat_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)


    with st.sidebar:
        st.subheader("MultiDocument Chat Introduction")

        documents = st.file_uploader(
            "Upload your PDFs here and click on Process.",
            type="pdf",
            accept_multiple_files=True,
        )
        if st.button("Process"):
            with st.spinner("Processing..."):
                # 1. get all documents
                texts = get_documents_texts(documents)
                # 2. get all chunks
                chunks = get_text_chunks(texts)
                # 3. create vectorstore
                vectorstore = get_vectorstore(chunks)
                # 4. create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)


login_auth(title="MultiDocument Chat",func=run_page)

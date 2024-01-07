import streamlit as st
import os
from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.retrievers.web_research import WebResearchRetriever
from utils import login_auth
import os

st.set_page_config(page_title="LLM Based Web Search", page_icon="🌐")

load_dotenv()
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
API_DEPLOYMENT_NAME = os.getenv("API_DEPLOYMENT_NAME")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")


def settings():
    # 配置EmbeddingModle
    from langchain.embeddings import AzureOpenAIEmbeddings

    embeddings_model = AzureOpenAIEmbeddings(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        openai_api_key=AZURE_OPENAI_API_KEY,
        azure_deployment="ada2",
        openai_api_version=OPENAI_API_VERSION,
    )

    # 配置向量数据库
    import chromadb
    from langchain_community.vectorstores import Chroma

    chroma_client = chromadb.PersistentClient()

    chroma_collection_name = "langchain_chroma_default"
    chroma_client.get_or_create_collection(chroma_collection_name)

    vectorstore_public = Chroma(
        client=chroma_client,
        collection_name=chroma_collection_name,
        embedding_function=embeddings_model,
    )

    # 配置LLM
    from langchain.chat_models import AzureChatOpenAI

    llm = AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        openai_api_version=OPENAI_API_VERSION,
        openai_api_key=AZURE_OPENAI_API_KEY,
        deployment_name=API_DEPLOYMENT_NAME,
        temperature=0,
        streaming=True,
    )

    # Search
    from langchain.utilities import GoogleSearchAPIWrapper

    search = GoogleSearchAPIWrapper(
        google_api_key=GOOGLE_API_KEY,
        google_cse_id=GOOGLE_CSE_ID,
    )

    # Initialize
    web_retriever = WebResearchRetriever.from_llm(
        vectorstore=vectorstore_public, llm=llm, search=search, num_search_results=1
    )

    return web_retriever, llm


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.info(self.text)


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container.expander("Context Retrieval")

    def on_retriever_start(self, query: str, **kwargs):
        self.container.write(f"**Question:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        # self.container.write(documents)
        for idx, doc in enumerate(documents):
            source = doc.metadata["source"]
            self.container.write(f"**Results from {source}**")
            self.container.text(doc.page_content)


def run_page():

    # st.sidebar.image("img/ai.png")
    st.header("`Interweb Explorer`")
    st.info(
        "`I am an AI that can answer questions by exploring, reading, and summarizing web pages."
        "I can be configured to use different modes: public API or private (no data sharing).`"
    )

    # Make retriever and llm
    if "retriever" not in st.session_state:
        st.session_state["retriever"], st.session_state["llm"] = settings()
    web_retriever = st.session_state.retriever
    llm = st.session_state.llm

    # User input
    question = st.text_input("`Ask a question:`")

    if question:
        # Generate answer (w/ citations)
        import logging

        logging.basicConfig()
        logging.getLogger("langchain.retrievers.web_research").setLevel(logging.INFO)
        qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm, chain_type="stuff", retriever=web_retriever
        )

        # Write answer and sources
        retrieval_streamer_cb = PrintRetrievalHandler(st.container())
        answer = st.empty()
        stream_handler = StreamHandler(answer, initial_text="`Answer:`\n\n")
        result = qa_chain(
            {"question": question}, callbacks=[retrieval_streamer_cb, stream_handler]
        )
        answer.info("`Answer:`\n\n" + result["answer"])
        st.info("`Sources:`\n\n" + result["sources"])


login_auth(title="`Interweb Explorer`",func=run_page)
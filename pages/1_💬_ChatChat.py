import streamlit as st
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage

import os
from dotenv import load_dotenv
load_dotenv()


AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
OPENAI_API_VERSION = os.getenv('OPENAI_API_VERSION')
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
API_DEPLOYMENT_NAME = os.getenv('API_DEPLOYMENT_NAME')

st.set_page_config(page_title="💬 ChatChat")
st.title("💬 ChatChat")

with st.sidebar:
    st.title("ChatChat Introduction")
    st.markdown(
        """
        Chat with your AI Assistant.
        """
    )
    # if openai_api_key := st.text_input("OpenAI-API-Key", type="password"):
    #     st.secrets.openai_api_key = openai_api_key

# 初始化ChatModel
if "chat_model" not in st.session_state:
    st.session_state.chat_model = AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        openai_api_version=OPENAI_API_VERSION,
        openai_api_key=AZURE_OPENAI_API_KEY,
        deployment_name=API_DEPLOYMENT_NAME,
        temperature=0,
        streaming=True,
    )
chat_model = st.session_state.chat_model

# 初始化Messages
if "messages" not in st.session_state:
    chatbot_system_prompt = SystemMessage(content="You are a nice chatbot having a conversation with a human.")
    chatbot_init_messages = AIMessage(content="Hello! How can I assist you today?")
    st.session_state.messages = [chatbot_system_prompt, chatbot_init_messages]

# 展示message
for message in st.session_state.messages:
    if message.type not in ("system"):
        with st.chat_message(message.type):
            st.markdown(message.content)

if prompt := st.chat_input("What's up?"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append(HumanMessage(content=prompt))

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in chat_model.stream(st.session_state.messages):
            full_response += response.content
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append(AIMessage(content=full_response))

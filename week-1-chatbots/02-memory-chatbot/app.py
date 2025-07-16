import streamlit as st
from dotenv import load_dotenv
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Load environment variables
load_dotenv()
azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# Streamlit UI setup
st.set_page_config(page_title="🧠 GPT Memory Chatbot", page_icon="💬")
st.title("💬 GPT Chatbot with Memory")
st.write("Talk with an AI that remembers your conversation!")

# Initialize memory only once per session
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

# Initialize LLM and chain
llm = AzureChatOpenAI(
    openai_api_version="2024-02-15-preview",
    azure_endpoint=azure_endpoint,
    azure_deployment=azure_deployment,
    api_key=azure_api_key,
    streaming=True
)

conversation = ConversationChain(
    llm=llm,
    memory=st.session_state.memory,
    verbose=False
)

# Get user input
user_input = st.chat_input("Say something...")

if user_input:
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get model response using conversation chain
    with st.chat_message("assistant"):
        response = conversation.predict(input=user_input)
        st.markdown(response)

# Optional: Show memory content (for debugging)
# with st.expander("🧠 Memory Buffer"):
#     st.write(st.session_state.memory.buffer)

import streamlit as st
from dotenv import load_dotenv
import os
import fitz  # PyMuPDF

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI

# Load keys
load_dotenv()
azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# GPT setup
llm = AzureChatOpenAI(
    openai_api_version="2024-02-15-preview",
    azure_endpoint=azure_endpoint,
    azure_deployment=azure_deployment,
    api_key=azure_api_key,
    streaming=True
)

print("✅ App started")
st.write("🚀 App is running!")
# Streamlit UI
st.set_page_config(page_title="📄 Resume & PDF Reviewer", page_icon="📄")
st.title("📄 GPT Resume / PDF Analyzer")
st.write("Upload any PDF and ask questions about its content.")

# Upload PDF
pdf = st.file_uploader("Upload a PDF file", type=["pdf"])

if pdf:
    # Read text
    with fitz.open(stream=pdf.read(), filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()

    st.success("✅ PDF Loaded Successfully!")

    user_query = st.text_input("Ask a question about the PDF:")

    if user_query:
        # Prompt Template
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant analyzing this document:\n{text}"),
            ("user", "{question}")
        ])

        chain = prompt | llm | StrOutputParser()

        with st.spinner("Analyzing..."):
            response = chain.invoke({"text": text, "question": user_query})
            st.success("Answer:")
            st.write(response)

import os
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langchain_groq import ChatGroq


# -------------------------
# CONFIG
# -------------------------

DOCS_PATH = "docs"
ACCESS_CODE = "company123"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# -------------------------
# LOAD VECTOR STORE
# -------------------------

@st.cache_resource
def load_vector_store():

    documents = []

    for file in os.listdir(DOCS_PATH):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DOCS_PATH, file))
            documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(docs, embeddings)

    return vectorstore


# -------------------------
# LOAD LLM
# -------------------------

def load_llm():

    return ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.3-70b-versatile",
        temperature=0
    )


# -------------------------
# STREAMLIT UI
# -------------------------

st.set_page_config(page_title="Company Knowledge Chatbot")

st.title("📚 Company Knowledge Chatbot")

st.sidebar.title("Access")

access = st.sidebar.text_input(
    "Company Access Code",
    type="password"
)

if access != ACCESS_CODE:
    st.warning("Enter the company access code")
    st.stop()


vectorstore = load_vector_store()
retriever = vectorstore.as_retriever()

llm = load_llm()


prompt = ChatPromptTemplate.from_template(
"""
Answer the question using the context below.

Context:
{context}

Question:
{question}
"""
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
)


if "messages" not in st.session_state:
    st.session_state.messages = []


query = st.chat_input("Ask about the documents")

if query:

    response = rag_chain.invoke(query)

    st.session_state.messages.append(
        {"user": query, "bot": response.content}
    )


for msg in st.session_state.messages:

    with st.chat_message("user"):
        st.write(msg["user"])

    with st.chat_message("assistant"):
        st.write(msg["bot"])
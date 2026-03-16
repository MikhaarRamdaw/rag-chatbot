import os
import streamlit as st
from PIL import Image

# ------------------------------
# PAGE CONFIG (MUST BE FIRST)
# ------------------------------

st.set_page_config(
    page_title="AI-Sport Knowledge Assistant",
    layout="wide"
)

# ------------------------------
# IMPORTS
# ------------------------------

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langchain_groq import ChatGroq


# ------------------------------
# CONFIG
# ------------------------------

DOCS_PATH = "docs"
ACCESS_CODE = "company123"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ------------------------------
# LOAD LOGO
# ------------------------------

LOGO = None

if os.path.exists("logo.png"):
    try:
        LOGO = Image.open("logo.png")
    except:
        LOGO = None


# ------------------------------
# LOGIN SYSTEM
# ------------------------------

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False


def login():

    st.title("🔐 AI-Sport Secure Access")

    st.markdown("Enter your company access code to continue.")

    access_code = st.text_input("Company Access Code", type="password")

    if st.button("Login"):

        if access_code == ACCESS_CODE:
            st.session_state.authenticated = True
            st.rerun()

        else:
            st.error("Invalid access code")


if not st.session_state.authenticated:
    login()
    st.stop()


# ------------------------------
# HEADER / BRANDING
# ------------------------------

col1, col2 = st.columns([1, 5])

with col1:
    if LOGO:
        st.image(LOGO, width=120)

with col2:
    st.title("AI-Sport Knowledge Assistant")
    st.caption("AI-Sport Internal Support Chatbot")

st.divider()


# ------------------------------
# LOAD VECTOR STORE
# ------------------------------

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


# ------------------------------
# LOAD LLM
# ------------------------------

def load_llm():

    return ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.3-70b-versatile",
        temperature=0
    )


# ------------------------------
# LOAD COMPONENTS
# ------------------------------

vectorstore = load_vector_store()
retriever = vectorstore.as_retriever()

llm = load_llm()


# ------------------------------
# PROMPT TEMPLATE
# ------------------------------

prompt = ChatPromptTemplate.from_template(
"""
You are AI-Sport, the internal company support assistant.

Answer ONLY using the provided documentation.

If the answer cannot be found in the documents, respond with:

"Please consult with your regional head operator for further assistance."

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


# ------------------------------
# CHAT MEMORY
# ------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# ------------------------------
# DISPLAY CHAT HISTORY
# ------------------------------

for message in st.session_state.messages:

    if message["role"] == "assistant":

        with st.chat_message("assistant", avatar=LOGO):
            st.markdown(message["content"])

    else:

        with st.chat_message("user"):
            st.markdown(message["content"])


# ------------------------------
# CHAT INPUT
# ------------------------------

query = st.chat_input("Ask AI-Sport about the system...")

if query:

    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.markdown(query)

    user_input = query.lower()

    # ------------------------------
    # SMALL TALK RESPONSES
    # ------------------------------

    small_talk = {
        "hi": "Hello! I'm AI-Sport, your internal support assistant. How can I help you today?",
        "hello": "Hello! How can I assist you with the AI-Sport system today?",
        "hey": "Hey there! What would you like help with?",
        "thanks": "You're welcome! Let me know if you need anything else.",
        "thank": "You're welcome! I'm here to help.",
        "how are you": "I'm running perfectly and ready to help with AI-Sport support questions.",
        "good morning": "Good morning! How can I assist you today?",
        "good afternoon": "Good afternoon! What can I help you with?",
        "good evening": "Good evening! How can I assist you today?"
    }

    answer = None

    for key in small_talk:
        if key in user_input:
            answer = small_talk[key]
            break

    with st.chat_message("assistant", avatar=LOGO):

        with st.spinner("AI-Sport is thinking..."):

            if answer is None:

                docs = retriever.invoke(query)

                if len(docs) == 0:

                    answer = "Please consult with your regional head operator for further assistance. I would hate to mislead/misguide you :)"

                else:

                    response = rag_chain.invoke(query)
                    answer = response.content

            st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
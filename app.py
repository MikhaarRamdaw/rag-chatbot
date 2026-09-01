import os
import io
import streamlit as st
from PIL import Image

# ------------------------------
# PAGE CONFIG (MUST BE FIRST)
# ------------------------------

st.set_page_config(
    page_title="AISA Knowledge Assistant",
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

# Access code and API key should live in Streamlit secrets (.streamlit/secrets.toml
# locally, or the "Secrets" panel on Streamlit Cloud), never hardcoded in source.
#
# .streamlit/secrets.toml should contain:
#   GROQ_API_KEY = "your-groq-key"
#   ACCESS_CODE = "your-access-code"
ACCESS_CODE = st.secrets.get("ACCESS_CODE", os.getenv("ACCESS_CODE", ""))
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))

# Chunks are not hard-filtered by a raw distance cutoff — the LLM decides
# relevance itself, guided by the prompt. This is how many candidate chunks it
# gets to look at per question.
RETRIEVAL_K = 8

# How many previous user/assistant turns to include as conversation context, so
# the assistant doesn't re-ask a question it already got an answer to.
MAX_HISTORY_TURNS = 4


# ------------------------------
# SYNONYM EXPANSION
# ------------------------------
# Widens the query with domain terms before it hits FAISS, so operator phrasing
# ("graphics not showing") matches doc phrasing ("no graphics", "overlay",
# "scoreboard") even when the exact words differ.

SYNONYM_MAP = {
    "graphics": ["overlay", "scoreboard", "score bug", "sss logo", "ads", "banner", "advance gfx", "sportvot"],
    "stream": ["video", "feed", "playback", "footage", "vod"],
    "vpu": ["box", "unit", "server", "the machine"],
    "chu": ["camera", "camera head"],
    "sound": ["audio", "mic", "microphone", "static", "buzzing", "humming"],
    "network": ["internet", "wifi", "connection", "router"],
    "timer": ["clock", "time", "sync"],
    "tracking": ["camera not following", "not panning", "not following play"],
}


def expand_query(query: str) -> str:
    lower = query.lower()
    extra_terms = set()
    for term, synonyms in SYNONYM_MAP.items():
        if term in lower or any(s in lower for s in synonyms):
            extra_terms.add(term)
            extra_terms.update(synonyms)
    if extra_terms:
        return query + " " + " ".join(sorted(extra_terms))
    return query


def build_retrieval_query(current_query: str, prior_messages: list) -> str:
    """Combine the current message with the last thing AISA said.

    Short replies like "S1" or "connection" are meaningless to a similarity
    search on their own — they only make sense as an answer to AISA's previous
    clarifying question. Prepending that question gives the retriever the
    actual topic to search for.
    """
    last_assistant = None
    for m in reversed(prior_messages):
        if m["role"] == "assistant":
            last_assistant = m["content"]
            break

    if last_assistant:
        combined = f"{last_assistant} {current_query}"
    else:
        combined = current_query

    return expand_query(combined)


def build_history_text(messages: list, max_turns: int = MAX_HISTORY_TURNS) -> str:
    """Render the last few turns as plain text so the LLM has conversational
    context and doesn't repeat a clarifying question it already asked."""
    trimmed = messages[-(max_turns * 2):]
    lines = []
    for m in trimmed:
        speaker = "Operator" if m["role"] == "user" else "AISA"
        lines.append(f"{speaker}: {m['content']}")
    return "\n".join(lines) if lines else "(no prior messages)"


# ------------------------------
# LOAD LOGO
# ------------------------------
# Handles the case where "logo.png" is actually a folder containing the real
# image file. Tries the direct path first, then falls back to looking for an
# image file one level inside it.

def _find_logo_path():
    if os.path.isfile("logo.png"):
        return "logo.png"
    if os.path.isdir("logo.png"):
        for name in os.listdir("logo.png"):
            if name.lower().endswith((".png", ".jpg", ".jpeg")):
                return os.path.join("logo.png", name)
    return None


LOGO = None

logo_path = _find_logo_path()

if logo_path:
    try:
        with open(logo_path, "rb") as f:
            LOGO = Image.open(io.BytesIO(f.read()))
            LOGO.load()
    except (FileNotFoundError, OSError, ValueError) as e:
        st.warning(f"Couldn't load {logo_path}: {e}")
        LOGO = None
elif os.path.exists("logo.png"):
    st.warning(
        "'logo.png' exists but is a folder with no image file inside it. "
        "Move the actual .png/.jpg file directly into the project root and "
        "name it logo.png, or place it inside the logo.png folder."
    )


# ------------------------------
# LOGIN SYSTEM
# ------------------------------

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False


def login():

    st.title("🔐 AISA Secure Access")

    st.markdown("Enter your company access code to continue.")

    access_code = st.text_input("Company Access Code", type="password")

    if st.button("Login"):

        if not ACCESS_CODE:
            st.error("No access code configured. Set ACCESS_CODE in Streamlit secrets.")
        elif access_code == ACCESS_CODE:
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
    st.title("AISA Knowledge Assistant")
    st.caption("AI Sport Africa Internal Support Chatbot")

st.divider()


# ------------------------------
# LOAD VECTOR STORE
# ------------------------------

@st.cache_resource
def load_vector_store():

    documents = []

    if not os.path.isdir(DOCS_PATH):
        st.error(f"Docs folder '{DOCS_PATH}' not found. Create it and add your PDFs.")
        st.stop()

    for file in os.listdir(DOCS_PATH):

        if file.endswith(".pdf"):

            loader = PyPDFLoader(os.path.join(DOCS_PATH, file))
            documents.extend(loader.load())

    if not documents:
        st.error(f"No PDFs found in '{DOCS_PATH}'. Add at least one PDF and restart.")
        st.stop()

    # Larger chunks + structure-aware separators so that problem/solution content
    # lands in the same chunk instead of being sliced apart mid-section.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1800,
        chunk_overlap=300,
        separators=["\n\n\n", "\n\n", "\n", ". ", " "]
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

    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY not set. Add it to Streamlit secrets.")
        st.stop()

    return ChatGroq(
        api_key=GROQ_API_KEY,
        # llama-3.3-70b-versatile was decommissioned by Groq (shutdown Aug 16, 2026).
        # openai/gpt-oss-120b is the recommended replacement; qwen/qwen3.6-27b also works.
        model="openai/gpt-oss-120b",
        temperature=0
    )


# ------------------------------
# LOAD COMPONENTS
# ------------------------------

vectorstore = load_vector_store()
llm = load_llm()


# ------------------------------
# PROMPT TEMPLATE
# ------------------------------

FALLBACK_MESSAGE = (
    "I am not trained to answer this question at the current time. "
    "Please do reach out to a head operator or raise a ticket with support "
    "to assist you further."
)

# Curated domain knowledge that's always available to the assistant, regardless of
# what the PDF retrieval finds. Add new entries here as you teach the bot more —
# each one should be a short, self-contained fact or definition.
DOMAIN_NOTES = """
- Tracking being off, inaccurate, or not tracking correctly is typically caused by
  one or more of: calibration being off, the CHU (camera head unit) having moved
  from its calibrated position, or people/objects in the field of play interfering
  with the tracking algorithm.
- "Graphics" refers to the overlay shown on the stream: adverts/ads, banners, and
  the scoreboard. The graphics provider is branded "Advance GFX" but is still
  referred to as "SportVot" inside the CMS (under VAS / Overlay Provider).
- To raise a support ticket, email support@aisport.africa with the system name and
  a description of the issue being faced.
""".strip()

prompt = ChatPromptTemplate.from_template(
"""
You are AISA, the internal expert assistant for AI Sport Africa's Pixellot camera
systems and Advance GFX/SportVot overlays. You've effectively configured hundreds
of installs and know the hardware, the software, and the common failure modes
cold. Answer the way a senior field technician would talk to a colleague: direct,
practical, no filler.

Conversation so far (most recent last):
{{history}}

The operator's latest message may be a short answer to a question YOU asked
earlier in this conversation (e.g. "S1", "connection"). Read the conversation
above before deciding what they mean. NEVER ask a clarifying question you have
already asked and already received an answer to in this conversation — use the
answer that was given and move the conversation forward. Only ask a NEW
clarifying question if something is still genuinely missing that hasn't been
asked yet.

Ground every factual claim in the context below, which combines curated domain
knowledge and excerpts from the documentation. The excerpts are the closest
matches found by search, but search isn't perfect — some may be irrelevant to
this specific question. Use only the ones that actually answer it, and ignore
the rest. If the context gives a partial answer, say what it covers and be
explicit about what it doesn't. If none of the excerpts and none of the domain
knowledge actually address the question (and there's no unanswered clarifying
question left to ask), respond with exactly:
"{fallback}"

When you do answer, prefer this shape where it fits:
1. A one-line diagnosis or direct answer.
2. Concrete steps or facts, in order.
3. What to check or do next if that doesn't resolve it.

Domain knowledge:
{domain_notes}

Documentation excerpts:
{{context}}

Operator's latest message:
{{question}}
""".format(fallback=FALLBACK_MESSAGE, domain_notes=DOMAIN_NOTES)
)


def format_docs(docs):
    if not docs:
        return "(no matching documentation excerpts found)"
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {
        "context": lambda x: format_docs(x["docs"]),
        "question": lambda x: x["question"],
        "history": lambda x: x["history"],
    }
    | prompt
    | llm
)


def retrieve_relevant(query, k=RETRIEVAL_K):
    """Return the k closest chunks. No hard distance cutoff — the LLM is
    instructed to only use excerpts that actually answer the question and to
    ignore the rest, so a slightly noisy top-k is safer than a brittle filter
    that sometimes silently discards everything useful."""
    results = vectorstore.similarity_search_with_score(query, k=k)
    return [doc for doc, score in results]


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

query = st.chat_input("Ask AISA about the system...")

if query:

    # Snapshot prior messages BEFORE appending the new one, so history/retrieval
    # helpers can distinguish "what's already been said" from "what's new".
    prior_messages = list(st.session_state.messages)

    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.markdown(query)

    user_input = query.lower().strip()

    # ------------------------------
    # SMALL TALK RESPONSES
    # ------------------------------
    # Matched only when the whole message IS the greeting/phrase (optionally with
    # trailing punctuation), not just contains it as a substring — otherwise words
    # like "graphics" (contains "hi") or "unthankful" (contains "thank") false-match.

    small_talk = {
        "hi": "Hello! I'm AISA, your internal support assistant. How can I help you today?",
        "hello": "Hello! How can I assist you with the AI Sport Africa system today?",
        "hey": "Hey there! What would you like help with?",
        "thanks": "You're welcome! Let me know if you need anything else.",
        "thank you": "You're welcome! I'm here to help.",
        "how are you": "I'm running perfectly and ready to help with AI Sport Africa support questions.",
        "good morning": "Good morning! How can I assist you today?",
        "good afternoon": "Good afternoon! What can I help you with?",
        "good evening": "Good evening! How can I assist you today?"
    }

    normalized_input = user_input.rstrip("!.?")

    answer = None

    for key in small_talk:
        if normalized_input == key:
            answer = small_talk[key]
            break

    with st.chat_message("assistant", avatar=LOGO):

        with st.spinner("AISA is thinking..."):

            if answer is None:

                retrieval_query = build_retrieval_query(query, prior_messages)
                docs = retrieve_relevant(retrieval_query)
                history_text = build_history_text(prior_messages)

                response = rag_chain.invoke({
                    "docs": docs,
                    "question": query,
                    "history": history_text,
                })
                answer = response.content

            st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
    
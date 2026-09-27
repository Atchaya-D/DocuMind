import os
import tempfile

import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="DocuMind",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🧠 DocuMind")
    st.caption("MULTI-MODAL DOCUMENT INTELLIGENCE")

    st.divider()

    st.subheader("Workspace")

    st.button("⌂  Overview", use_container_width=True)
    st.button("📄  Documents", use_container_width=True)
    st.button("💬  AI Chat", use_container_width=True)
    st.button("📊  Analytics", use_container_width=True)

    st.divider()

    st.subheader("AI Engine")

    st.caption("Language Model")
    st.write("Llama 3.2 3B")

    st.caption("Embeddings")
    st.write("MiniLM")

    st.caption("Vector Store")
    st.write("FAISS")

    st.success("● System Online")


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# =========================================================
# LOAD LLM
# =========================================================

@st.cache_resource
def load_llm():

    return ChatOllama(
        model="llama3.2:3b",
        temperature=0
    )


# =========================================================
# CREATE VECTOR DATABASE
# =========================================================

def create_vectorstore(uploaded_file):

    # Save uploaded PDF temporarily
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:

        temp_file.write(uploaded_file.getbuffer())
        pdf_path = temp_file.name

    # Load PDF
    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    # Add useful metadata
    for document in documents:

        document.metadata["source_file"] = uploaded_file.name

        document.metadata["page_number"] = (
            document.metadata.get("page", 0) + 1
        )

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120
    )

    chunks = splitter.split_documents(documents)

    # Embeddings
    embeddings = load_embeddings()

    # FAISS
    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    # Remove temporary file
    os.remove(pdf_path)

    return vectorstore, len(documents), len(chunks)


# =========================================================
# RAG ANSWER
# =========================================================

def ask_document(vectorstore, question):

    # Retrieve relevant chunks
    retrieved_docs = vectorstore.similarity_search(
        question,
        k=5
    )

    if not retrieved_docs:

        return "I couldn't find relevant information in the document.", []

    # Build context
    context_parts = []

    for doc in retrieved_docs:

        page = doc.metadata.get("page_number", "?")

        context_parts.append(
            f"[Page {page}]\n{doc.page_content}"
        )

    context = "\n\n".join(context_parts)

    # Prompt
    prompt = f"""
You are DocuMind, a document question-answering assistant.

Answer the user's question using ONLY the information
provided in the document context below.

If the answer is not present in the context, say:
"I could not find this information in the uploaded document."

Do not invent information.

Give a clear and concise answer.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    llm = load_llm()

    response = llm.invoke(prompt)

    answer = response.content

    # Collect sources
    sources = []

    for doc in retrieved_docs:

        page = doc.metadata.get("page_number", "?")

        source = {
            "file": doc.metadata.get(
                "source_file",
                "Unknown"
            ),
            "page": page
        }

        if source not in sources:
            sources.append(source)

    return answer, sources


# =========================================================
# MAIN PAGE
# =========================================================

st.caption("DOCUMENT INTELLIGENCE WORKSPACE")

st.title("Understand your documents with AI.")

st.write(
    "DocuMind helps you search, analyze and understand "
    "complex documents using Retrieval-Augmented Generation."
)

st.info(
    "📄 Upload a PDF → 🔎 Ask questions → 💡 Get grounded answers with sources."
)

st.divider()


# =========================================================
# CAPABILITIES
# =========================================================

st.header("Capabilities")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.info("📄")
    st.subheader("Text Intelligence")
    st.write("Search and understand document content.")

with c2:
    st.info("📊")
    st.subheader("Table Analysis")
    st.write("Extract and query structured information.")

with c3:
    st.info("🖼️")
    st.subheader("Visual Retrieval")
    st.write("Find figures, charts and document images.")

with c4:
    st.info("✓")
    st.subheader("Grounded Answers")
    st.write("Trace answers back to document sources.")


st.divider()


# =========================================================
# UPLOAD
# =========================================================

st.header("Upload a document")

st.write(
    "Upload a PDF to create your document knowledge base."
)

uploaded_file = st.file_uploader(
    "Choose a PDF document",
    type=["pdf"],
    accept_multiple_files=False
)


# =========================================================
# PROCESS DOCUMENT
# =========================================================

if uploaded_file:

    # Only process if a new file was uploaded
    if (
        "current_file" not in st.session_state
        or st.session_state.current_file != uploaded_file.name
    ):

        with st.spinner(
            "Processing document... This may take a moment."
        ):

            vectorstore, pages, chunks = create_vectorstore(
                uploaded_file
            )

            st.session_state.vectorstore = vectorstore
            st.session_state.current_file = uploaded_file.name
            st.session_state.pages = pages
            st.session_state.chunks = chunks
            st.session_state.messages = []

        st.success(
            f"✓ {uploaded_file.name} processed successfully."
        )

    else:

        st.success(
            f"✓ {uploaded_file.name} is ready."
        )


# =========================================================
# DOCUMENT OVERVIEW
# =========================================================

if uploaded_file:

    st.subheader("Document Overview")

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric(
            "Status",
            "Ready"
        )

    with m2:
        st.metric(
            "Pages",
            st.session_state.get("pages", "—")
        )

    with m3:
        st.metric(
            "Chunks",
            st.session_state.get("chunks", "—")
        )

    with m4:
        st.metric(
            "Engine",
            "Llama 3.2"
        )


st.divider()


# =========================================================
# DOCUMENT ASSISTANT
# =========================================================

st.header("💬 Document Assistant")

st.write(
    "Ask questions about your uploaded document."
)


# Display previous messages
if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.write(message["content"])

        if message.get("sources"):

            st.caption("📚 Sources")

            for source in message["sources"]:

                st.caption(
                    f"• {source['file']} — Page {source['page']}"
                )


# Chat input
if not uploaded_file:

    st.info(
        "⬆️ Upload a document above to start asking questions."
    )

else:

    question = st.chat_input(
        "Ask anything about your document..."
    )

    if question:

        # User message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):
            st.write(question)

        # Assistant
        with st.chat_message("assistant"):

            with st.spinner(
                "🔎 Searching document and generating answer..."
            ):

                try:

                    answer, sources = ask_document(
                        st.session_state.vectorstore,
                        question
                    )

                    st.write(answer)

                    st.caption("📚 Sources")

                    for source in sources:

                        st.caption(
                            f"• {source['file']} — Page {source['page']}"
                        )

                    # Save conversation
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        }
                    )

                except Exception as e:

                    st.error(
                        f"Something went wrong: {e}"
                    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "DocuMind • Multi-Modal Document Intelligence • "
    "Python • LangChain • FAISS • Ollama"
)
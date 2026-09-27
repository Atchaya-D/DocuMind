from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama


# ==========================================
# 1. FIND ALL PDF DOCUMENTS
# ==========================================

pdf_folder = Path("data/documents")

pdf_files = list(pdf_folder.glob("*.pdf"))

if not pdf_files:
    print("No PDF files found in data/documents")
    exit()

print(f"Found {len(pdf_files)} PDF document(s):")

for pdf in pdf_files:
    print("-", pdf.name)


# ==========================================
# 2. LOAD ALL PDFs
# ==========================================

all_documents = []

for pdf_file in pdf_files:

    print(f"\nLoading: {pdf_file.name}")

    loader = PyPDFLoader(str(pdf_file))

    documents = loader.load()

    # Add source filename to metadata
    for document in documents:
        document.metadata["source_file"] = pdf_file.name

    all_documents.extend(documents)


print("\nTotal pages loaded:", len(all_documents))


# ==========================================
# 3. REMOVE EMPTY / SHORT PAGES
# ==========================================

all_documents = [
    doc for doc in all_documents
    if len(doc.page_content.strip()) > 100
]

print("Usable pages:", len(all_documents))


# ==========================================
# 4. SPLIT INTO CHUNKS
# ==========================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=150,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]
)

chunks = text_splitter.split_documents(all_documents)

print("Total chunks:", len(chunks))


# ==========================================
# 5. CREATE EMBEDDINGS
# ==========================================

print("\nCreating embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ==========================================
# 6. CREATE FAISS DATABASE
# ==========================================

vector_store = FAISS.from_documents(
    chunks,
    embeddings
)

print("Multi-document vector database created!")


# ==========================================
# 7. LOAD LLM
# ==========================================

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


# ==========================================
# 8. ASK QUESTION
# ==========================================

query = input("\nAsk a question about your documents: ")


# ==========================================
# 9. RETRIEVE RELEVANT CHUNKS
# ==========================================

results = vector_store.similarity_search(
    query,
    k=8
)


# ==========================================
# 10. FILTER UNWANTED RESULTS
# ==========================================

filtered_results = []

for result in results:

    text = result.page_content.lower()

    if "acknowledgement" in text[:100]:
        continue

    if "declaration" in text[:100]:
        continue

    filtered_results.append(result)


results = filtered_results[:5]


# ==========================================
# 11. BUILD CONTEXT
# ==========================================

context_parts = []

for result in results:

    page = result.metadata.get("page", "Unknown")

    source_file = result.metadata.get(
        "source_file",
        "Unknown document"
    )

    # Convert zero-based page index
    display_page = (
        page + 1
        if isinstance(page, int)
        else page
    )

    context_parts.append(
        f"""
[Document: {source_file}]
[Page: {display_page}]

{result.page_content}
"""
    )

context = "\n\n".join(context_parts)


# ==========================================
# 12. CREATE RAG PROMPT
# ==========================================

prompt = f"""
You are a multi-document question-answering assistant.

Answer the user's question using ONLY the document
evidence provided below.

Do NOT use outside knowledge.

If the answer is not available in the evidence, say:

"The answer is not available in the retrieved documents."

Important instructions:

1. Do not invent information.
2. Do not make assumptions.
3. Identify which document supports the answer.
4. Mention the page number when possible.
5. If multiple documents are relevant, compare them clearly.
6. Do not mention irrelevant documents.

DOCUMENT EVIDENCE:

{context}

USER QUESTION:

{query}
"""


# ==========================================
# 13. GENERATE ANSWER
# ==========================================

print("\nGenerating answer...\n")

response = llm.invoke(prompt)


# ==========================================
# 14. DISPLAY ANSWER
# ==========================================

print("=" * 60)
print("ANSWER")
print("=" * 60)

print(response.content)


# ==========================================
# 15. DISPLAY SOURCES
# ==========================================

print("\n" + "=" * 60)
print("SOURCES")
print("=" * 60)

for result in results:

    page = result.metadata.get("page", "Unknown")

    source_file = result.metadata.get(
        "source_file",
        "Unknown document"
    )

    display_page = (
        page + 1
        if isinstance(page, int)
        else page
    )

    print(
        f"{source_file} | Page {display_page}"
    )
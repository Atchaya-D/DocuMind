from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama


# ==========================================
# 1. LOAD PDF
# ==========================================

loader = PyPDFLoader("data/documents/research_paper.pdf")
documents = loader.load()

print("Number of pages:", len(documents))


# ==========================================
# 2. REMOVE EMPTY / SHORT PAGES
# ==========================================

documents = [
    doc for doc in documents
    if len(doc.page_content.strip()) > 100
]

print("Usable pages:", len(documents))


# ==========================================
# 3. SPLIT INTO CHUNKS
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

chunks = text_splitter.split_documents(documents)

print("Number of chunks:", len(chunks))


# ==========================================
# 4. CREATE EMBEDDINGS
# ==========================================

print("\nCreating embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ==========================================
# 5. CREATE VECTOR DATABASE
# ==========================================

vector_store = FAISS.from_documents(
    chunks,
    embeddings
)

print("Vector database created successfully!")


# ==========================================
# 6. LOAD LOCAL LLM
# ==========================================

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


# ==========================================
# 7. ASK QUESTION
# ==========================================

query = input("\nAsk a question about the document: ")


# ==========================================
# 8. RETRIEVE WITH SIMILARITY SCORES
# ==========================================

results_with_scores = vector_store.similarity_search_with_score(
    query,
    k=8
)


# ==========================================
# 9. FILTER RESULTS
# ==========================================

filtered_results = []

for result, score in results_with_scores:

    text = result.page_content.lower()

    # Ignore obvious front-matter sections
    if "contents" in text[:100]:
        continue

    if "acknowledgement" in text[:100]:
        continue

    if "declaration" in text[:100]:
        continue

    # Keep only reasonably relevant results
    if score < 1.8:
        filtered_results.append((result, score))


# Keep maximum 5 results
filtered_results = filtered_results[:5]


# ==========================================
# 10. CHECK IF EVIDENCE EXISTS
# ==========================================

if not filtered_results:

    print("\nNo sufficiently relevant evidence was found.")

    print(
        "\nThe system could not find enough relevant "
        "information to answer the question."
    )

    exit()


# ==========================================
# 11. BUILD CONTEXT
# ==========================================

context_parts = []

for result, score in filtered_results:

    page = result.metadata.get("page", "Unknown")

    # Convert zero-based PDF page index to human page number
    display_page = page + 1 if isinstance(page, int) else page

    context_parts.append(
        f"[Page {display_page}]\n{result.page_content}"
    )

context = "\n\n".join(context_parts)


# ==========================================
# 12. RAG PROMPT
# ==========================================

prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the document
evidence provided below.

Do NOT use outside knowledge.

If the answer cannot be found in the evidence,
say exactly:

"The answer is not available in the retrieved document evidence."

Give a clear and concise answer.

Important instructions:

1. Do not invent information.
2. Do not make assumptions.
3. Cite the page number supporting the answer.
4. If multiple pages support the answer, mention all relevant pages.
5. Do not mention irrelevant pages.

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

print("=" * 50)
print("ANSWER")
print("=" * 50)

print(response.content)


# ==========================================
# 15. DISPLAY EVIDENCE
# ==========================================

print("\n" + "=" * 50)
print("EVIDENCE")
print("=" * 50)

for result, score in filtered_results:

    page = result.metadata.get("page", "Unknown")

    display_page = page + 1 if isinstance(page, int) else page

    print(f"\nPage {display_page}")
    print(f"Similarity score: {score:.4f}")
    print("-" * 40)
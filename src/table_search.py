import re

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from table_rag import load_table_documents


# Load tables
table_documents = load_table_documents()

print("Table documents loaded:", len(table_documents))


# Create embeddings
print("\nCreating table embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

table_vector_store = FAISS.from_documents(
    table_documents,
    embeddings
)

print("Table vector database created!")


# Ask question
query = input("\nAsk a question about the tables: ")

query_lower = query.lower()


# Detect table number
table_match = re.search(r"table\s*(\d+)", query_lower)

# Detect page number
page_match = re.search(r"page\s*(\d+)", query_lower)


# --------------------------------------------------
# CASE 1: Table number + page number
# --------------------------------------------------

if table_match and page_match:

    requested_table = int(table_match.group(1))
    requested_page = int(page_match.group(1))

    print(
        f"\nExact request: "
        f"Table {requested_table} on page {requested_page}"
    )

    results = [
        doc
        for doc in table_documents
        if doc.metadata["table_number"] == requested_table
        and doc.metadata["page"] == requested_page
    ]

    if not results:
        print("\nNo exact table found.")
        print("Using semantic search instead...")

        results = table_vector_store.similarity_search(
            query,
            k=3
        )


# --------------------------------------------------
# CASE 2: Only table number
# --------------------------------------------------

elif table_match:

    requested_table = int(table_match.group(1))

    print(f"\nExact table requested: Table {requested_table}")

    results = [
        doc
        for doc in table_documents
        if doc.metadata["table_number"] == requested_table
    ]

    if not results:
        print("\nExact table not found.")
        print("Using semantic search instead...")

        results = table_vector_store.similarity_search(
            query,
            k=3
        )


# --------------------------------------------------
# CASE 3: No table number
# --------------------------------------------------

else:

    results = table_vector_store.similarity_search(
        query,
        k=3
    )


# --------------------------------------------------
# Display results
# --------------------------------------------------

print("\n" + "=" * 60)
print("RELEVANT TABLES")
print("=" * 60)


for i, result in enumerate(results[:3], start=1):

    print(f"\nResult {i}")
    print("-" * 40)

    print(result.page_content)

    print("\nSource:")

    print(
        f"{result.metadata['source_file']} | "
        f"Page {result.metadata['page']} | "
        f"Table {result.metadata['table_number']}"
    )
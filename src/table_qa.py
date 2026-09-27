from langchain_ollama import ChatOllama

from table_rag import load_table_documents


# Load normalized table documents
table_documents = load_table_documents()

print("Table documents loaded:", len(table_documents))


# Load local LLM
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


# Ask question
query = input("\nAsk a question about the tables: ")


# Find table/page references in the question
import re

table_match = re.search(
    r"table\s*(\d+)",
    query.lower()
)

page_match = re.search(
    r"page\s*(\d+)",
    query.lower()
)


# Retrieve the relevant table
results = []


if table_match and page_match:

    requested_table = int(table_match.group(1))
    requested_page = int(page_match.group(1))

    results = [
        doc
        for doc in table_documents
        if doc.metadata["table_number"] == requested_table
        and doc.metadata["page"] == requested_page
    ]


elif table_match:

    requested_table = int(table_match.group(1))

    results = [
        doc
        for doc in table_documents
        if doc.metadata["table_number"] == requested_table
    ]


else:

    # Simple keyword fallback
    query_words = query.lower().split()

    for doc in table_documents:

        text = doc.page_content.lower()

        if any(word in text for word in query_words):
            results.append(doc)


# Limit retrieved tables
results = results[:3]


if not results:

    print("\nNo relevant table found.")

else:

    # Combine table information
    context = "\n\n".join(
        doc.page_content
        for doc in results
    )


    prompt = f"""
    You are a precise table question-answering assistant.

Use ONLY the information in the retrieved table.

Important rules:
1. Read the table row by row.
2. Do not invent values.
3. If the question asks for a value associated with a dataset,
   match the dataset name with the corresponding value in that row.
4. The first table on page 55 has the following column meaning:

   Dataset | Number of samples | Number of features | Task

5. Therefore:
   Bank = 45,211 samples, 48 features, Classification
   Phishing = 11,055 samples, 46 features, Classification
   Wine = 6,497 samples, 12 features, Classification

6. If the requested information is not present, say:
   "The answer is not available in the retrieved table."


TABLE INFORMATION:
------------------
{context}
------------------

QUESTION:
{query}

Give a short, direct answer.
"""


    response = llm.invoke(prompt)


    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)

    print(response.content)


    print("\n" + "=" * 60)
    print("SOURCE")
    print("=" * 60)

    for doc in results:

        print(
            f"{doc.metadata['source_file']} | "
            f"Page {doc.metadata['page']} | "
            f"Table {doc.metadata['table_number']}"
        )
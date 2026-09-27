from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader


def load_pdf(pdf_path):
    """
    Load a PDF and attach useful metadata
    to every page.
    """

    pdf_path = Path(pdf_path)

    loader = PyPDFLoader(str(pdf_path))

    documents = loader.load()

    for document in documents:

        document.metadata["source_file"] = pdf_path.name

        document.metadata["page_number"] = (
            document.metadata.get("page", 0) + 1
        )

    return documents


if __name__ == "__main__":

    pdf_path = "data/documents/research_paper.pdf"

    documents = load_pdf(pdf_path)

    print("Document:", pdf_path)
    print("Total pages:", len(documents))

    print("\nFirst page metadata:")
    print(documents[0].metadata)

    print("\nFirst page text preview:")
    print(documents[0].page_content[:500])
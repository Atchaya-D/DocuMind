import json

from langchain_core.documents import Document


TABLE_PATH = "data/extracted_tables.json"


def flatten_cell(cell):
    """
    Convert a cell into a list of clean values.
    """
    if cell is None:
        return []

    if isinstance(cell, list):
        return [
            str(value).strip()
            for value in cell
            if str(value).strip()
        ]

    value = str(cell).strip()

    if not value:
        return []

    return [value]


def normalize_table(rows):
    """
    Convert extracted PDF table cells into a
    simple row/column representation.
    """

    normalized = []

    for row in rows:

        # Collect all values from each cell
        expanded_row = [
            flatten_cell(cell)
            for cell in row
        ]

        # Find the maximum number of values in a cell
        max_values = max(
            (len(cell) for cell in expanded_row),
            default=1
        )

        # If cells contain multiple values,
        # create separate rows.
        for i in range(max_values):

            new_row = []

            for cell in expanded_row:

                if i < len(cell):
                    new_row.append(cell[i])
                else:
                    new_row.append("")

            normalized.append(new_row)

    return normalized


def load_table_documents():

    with open(
        TABLE_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        tables = json.load(file)

    documents = []

    for table in tables:

        source_file = table["source_file"]
        page = table["page"]
        table_number = table["table_number"]

        raw_rows = table["rows"]

        # Normalize extracted table
        rows = normalize_table(raw_rows)

        # Convert rows into readable text
        table_lines = []

        for row in rows:

            cleaned_row = [
                str(cell).strip()
                for cell in row
            ]

            table_lines.append(
                " | ".join(cleaned_row)
            )

        table_text = "\n".join(table_lines)

        document = Document(
            page_content=(
                f"Table {table_number} "
                f"from {source_file}, page {page}\n\n"
                f"{table_text}"
            ),

            metadata={
                "source_file": source_file,
                "page": page,
                "table_number": table_number,
                "content_type": "table"
            }
        )

        documents.append(document)

    return documents


if __name__ == "__main__":

    documents = load_table_documents()

    print("Table documents:", len(documents))

    print("\n========== FIRST TABLE ==========\n")

    print(documents[0].page_content)

    print("\n========== METADATA ==========\n")

    print(documents[0].metadata)
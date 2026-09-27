import fitz
import json
from pathlib import Path


PDF_PATH = "data/documents/research_paper.pdf"
OUTPUT_PATH = "data/extracted_tables.json"


def clean_cell(cell):

    if cell is None:
        return ""

    # Convert to string
    cell = str(cell)

    # Remove unnecessary spaces around lines
    lines = [
        line.strip()
        for line in cell.split("\n")
        if line.strip()
    ]

    return lines


def extract_tables(pdf_path):

    pdf = fitz.open(pdf_path)

    all_tables = []

    for page_number, page in enumerate(pdf, start=1):

        try:
            tables = page.find_tables()

            for table_number, table in enumerate(
                tables.tables,
                start=1
            ):

                data = table.extract()

                # Remove completely empty rows
                cleaned_data = []

                for row in data:

                    if not row:
                        continue

                    cleaned_row = []

                    for cell in row:

                        values = clean_cell(cell)

                        # Keep the values together for now
                        cleaned_row.append(values)

                    # Ignore completely empty rows
                    if any(cleaned_row):
                        cleaned_data.append(cleaned_row)

                if cleaned_data:

                    table_info = {
                        "source_file": Path(pdf_path).name,
                        "page": page_number,
                        "table_number": table_number,
                        "rows": cleaned_data
                    }

                    all_tables.append(table_info)

        except Exception as e:

            print(
                f"Could not process page {page_number}: {e}"
            )

    return all_tables


if __name__ == "__main__":

    print("Extracting tables...")

    tables = extract_tables(PDF_PATH)

    Path("data").mkdir(exist_ok=True)

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            tables,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\n==============================")
    print("TABLE EXTRACTION COMPLETE")
    print("==============================")

    print("Total tables:", len(tables))

    print("Saved to:")
    print(OUTPUT_PATH)

    # Display preview
    for table in tables:

        print(
            f"\nPage {table['page']} "
            f"| Table {table['table_number']}"
        )

        for row in table["rows"][:5]:
            print(row)
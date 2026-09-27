import fitz
import json
from pathlib import Path


PDF_PATH = "data/documents/research_paper.pdf"
IMAGE_DIR = Path("data/images")
OUTPUT_PATH = "data/extracted_images.json"


def extract_images(pdf_path):

    pdf = fitz.open(pdf_path)

    IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    images = []

    for page_number, page in enumerate(pdf, start=1):

        page_images = page.get_images(full=True)

        for image_number, image_info in enumerate(
            page_images,
            start=1
        ):

            xref = image_info[0]

            try:

                image_data = pdf.extract_image(xref)

                width = image_data["width"]
                height = image_data["height"]

                # Ignore extremely small images/logos
                if width < 100 or height < 100:
                    continue

                extension = image_data["ext"]

                image_path = (
                    IMAGE_DIR
                    / f"page_{page_number}_image_{image_number}.{extension}"
                )

                with open(image_path, "wb") as file:
                    file.write(image_data["image"])

                images.append({
                    "source_file": Path(pdf_path).name,
                    "page": page_number,
                    "image_number": image_number,
                    "width": width,
                    "height": height,
                    "image_path": str(image_path)
                })

            except Exception as e:

                print(
                    f"Could not process image "
                    f"on page {page_number}: {e}"
                )

    return images


if __name__ == "__main__":

    print("Extracting images...")

    images = extract_images(PDF_PATH)

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            images,
            file,
            indent=4
        )

    print("\n==============================")
    print("IMAGE EXTRACTION COMPLETE")
    print("==============================")

    print("Total images:", len(images))

    print("\nSaved to:")
    print(OUTPUT_PATH)

    for image in images:

        print(
            f"Page {image['page']} | "
            f"Image {image['image_number']} | "
            f"{image['width']}x{image['height']}"
        )
import json
import re
from pathlib import Path


IMAGE_METADATA = "data/extracted_images.json"


# Words that indicate the user is asking about visual content
VISUAL_KEYWORDS = {
    "image",
    "figure",
    "fig",
    "graph",
    "chart",
    "plot",
    "diagram",
    "visual",
    "picture",
    "illustration",
}


def load_images():
    """Load extracted image metadata."""

    with open(
        IMAGE_METADATA,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def extract_page_number(query):
    """Extract page number from a query."""

    match = re.search(
        r"\bpage\s*(\d+)\b",
        query.lower()
    )

    if match:
        return int(match.group(1))

    return None


def extract_image_number(query):
    """Extract image/figure number from a query."""

    patterns = [
        r"\bimage\s*(\d+)\b",
        r"\bfigure\s*(\d+)\b",
        r"\bfig\.?\s*(\d+)\b"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            query.lower()
        )

        if match:
            return int(match.group(1))

    return None


def is_visual_query(query):
    """Check whether the query is asking about visual content."""

    query_words = set(
        re.findall(
            r"\b[a-zA-Z]+\b",
            query.lower()
        )
    )

    return bool(
        query_words.intersection(
            VISUAL_KEYWORDS
        )
    )


def search_images(query, top_k=3):

    images = load_images()

    query_lower = query.lower()

    page_number = extract_page_number(
        query
    )

    image_number = extract_image_number(
        query
    )

    visual_query = is_visual_query(
        query
    )

    results = []

    for image in images:

        score = 0

        image_page = image["page"]
        image_num = image["image_number"]

        # --------------------------------
        # Exact page match
        # --------------------------------

        if page_number is not None:

            if image_page == page_number:
                score += 10
            else:
                score -= 5

        # --------------------------------
        # Exact image match
        # --------------------------------

        if image_number is not None:

            if image_num == image_number:
                score += 8

        # --------------------------------
        # Visual keywords
        # --------------------------------

        if visual_query:
            score += 3

        # --------------------------------
        # Specific page + image query
        # --------------------------------

        if (
            page_number is not None
            and image_number is not None
        ):

            if (
                image_page == page_number
                and image_num == image_number
            ):
                score += 20

        # --------------------------------
        # Query contains graph/chart/etc.
        # --------------------------------

        visual_terms = [
            word
            for word in VISUAL_KEYWORDS
            if word in query_lower
        ]

        if visual_terms:
            score += 2

        results.append(
            (score, image)
        )

    # Highest score first
    results.sort(
        key=lambda x: x[0],
        reverse=True
    )

    # Remove zero/negative matches
    filtered_results = [
        image
        for score, image in results
        if score > 0
    ]

    return filtered_results[:top_k]


def print_results(results):

    print(
        "\n========== IMAGE RESULTS ==========\n"
    )

    if not results:

        print(
            "No relevant images found."
        )

        return

    for image in results:

        print(
            f"Page {image['page']} | "
            f"Image {image['image_number']}"
        )

        print(
            f"Path: {image['image_path']}"
        )

        print(
            f"Size: "
            f"{image['width']}x"
            f"{image['height']}"
        )

        print(
            f"Source: "
            f"{image['source_file']}"
        )

        print()


if __name__ == "__main__":

    query = input(
        "Ask about an image: "
    )

    results = search_images(
        query,
        top_k=3
    )

    print_results(results)
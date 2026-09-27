import base64
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()


def encode_image(image_path):
    image_path = Path(image_path)

    with open(image_path, "rb") as file:
        image_bytes = file.read()

    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    if image_path.suffix.lower() == ".png":
        mime_type = "image/png"
    else:
        mime_type = "image/jpeg"

    return f"data:{mime_type};base64,{base64_image}"


def analyze_image(image_path, question):
    image_data = encode_image(image_path)

    response = client.responses.create(
        model="gpt-5.6-luna",
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": question
                    },
                    {
                        "type": "input_image",
                        "image_url": image_data,
                        "detail": "high"
                    }
                ]
            }
        ]
    )

    return response.output_text


if __name__ == "__main__":
    image_path = "data/images/page_60_image_1.png"

    question = """
Analyze this graph carefully.

Explain:
1. What type of graph it is.
2. What the x-axis represents.
3. What the y-axis represents.
4. What the main trend or comparison is.
5. Any important values, labels, or conclusions visible in the graph.

Do not invent values that cannot be read from the image.
"""

    print("\n========== VISION ANALYSIS ==========\n")

    answer = analyze_image(image_path, question)

    print(answer)
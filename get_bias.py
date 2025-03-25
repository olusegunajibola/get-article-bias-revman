# import fitz  # PyMuPDF
# import requests
# import json
# from groq import Groq
# import os
#
# # Constants
GROQ_API_KEY = "abcd"
#
# # client = Groq(api_key=os.getenv('GROQ_API_KEY'))
# client = Groq(api_key=GROQ_API_KEY)
#
# def extract_text_from_pdf(pdf_path):
#     """Extracts text from a PDF file."""
#     doc = fitz.open(pdf_path)
#     # text = ""
#     # for page in doc:
#     #     text += page.get_text("text") + "\n"
#     text = "\n".join([page.get_text("text") for page in doc])
#     return text
#
#
# def classify_bias(text):
#     """Sends extracted text to the Groq API for bias classification."""
#     prompt = f"""
#     Classify the biases judgement under one of the following criteria
#     A. low risk of bias
#     B. unclear risk of bias
#     C. high risk of bias
#
#     In addition, state a "support for judgement" for any of the class (A, B or C) chosen for the following:
#
#     1. Random sequence generation (selection bias)
#
#     Text:
#     {text}
#
#         {{
#             "classification": "<Low/Unclear/High> risk of bias",
#             "support": "<justification>"
#         }}
#     """
#     response  = client.chat.completions.create(
#         messages = [
#             {"role": "system", "content": "You are a journal review and classification assistant."},
#             {"role": "user", "content": prompt}
#         ],
#         model="deepseek-r1-distill-qwen-32b",
#         temperature=0.5,
#         max_completion_tokens=1024,
#     )
#     return response.choices[0].message.content
#
#
# # Main Execution
# if __name__ == "__main__":
#     pdf_path = "01_NEJMoa2034577.pdf"  # Change this to your PDF file path
#     text = extract_text_from_pdf(pdf_path)
#     result = classify_bias(text)
#
#     print("Bias Classification Result:")
#     print(result)

from groq import Groq
import fitz  # PyMuPDF

# Initialize Groq API Client
client = Groq(api_key=GROQ_API_KEY)

def extract_text_from_pdf(pdf_path):
    """Extract text from a given PDF file."""
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in doc])
    return text


def split_text(text, max_tokens=2000):
    """Splits text into smaller chunks to fit within token limits."""
    words = text.split()
    chunks = []
    chunk = []

    for word in words:
        chunk.append(word)
        if len(chunk) >= max_tokens:  # Approximate token count
            chunks.append(" ".join(chunk))
            chunk = []

    if chunk:
        chunks.append(" ".join(chunk))

    return chunks


def summarize_text(text):
    """Summarize the extracted text before classification to reduce token size."""
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are an AI that summarizes long text efficiently."},
            {"role": "user",
             "content": f"Summarize this text concisely while keeping key bias-related information:\n\n{text}"}
        ],
        model="llama-3.3-70b-versatile",  # Use a model optimized for summarization
        temperature=0.5,
        max_completion_tokens=512,
        top_p=1,
        stop=None,
        stream=False,
    )

    return chat_completion.choices[0].message.content


def classify_bias(text):
    """Classify bias using Groq API."""
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are an AI assistant trained to classify biases in text."},
            {"role": "user", "content": f"""
                Classify the biases judgment under one of the following criteria:
                A. Low risk of bias
                B. Unclear risk of bias
                C. High risk of bias

                In addition, provide a 'support for judgement' for any class (A, B, or C) chosen.

                Focus specifically on:
                1. Random sequence generation (selection bias)
                2. Allocation concealment (selection bias)
                3. Blinding of participants and personnel (performance bias)
                4. Blinding of outcome assessment (detection bias)
                5. Incomplete outcome data (attrition bias)
                6. Selective reporting (reporting bias)
                7. Other bias

                Text:
                {text}

                Respond in JSON format:
                {{
                    "classification": "<Low/Unclear/High> risk of bias",
                    "support": "<justification>"
                }}
            """}
        ],
        model="deepseek-r1-distill-qwen-32b",
        temperature=0.5,
        max_completion_tokens=512,  # Reduce to prevent token limit errors
        top_p=1,
        stop=None,
        stream=False,
    )

    return chat_completion.choices[0].message.content


# Main Execution
if __name__ == "__main__":
    pdf_path = "01_NEJMoa2034577.pdf"  # Change to your actual PDF path
    text = extract_text_from_pdf(pdf_path)

    chunks = split_text(text)  # Split into smaller chunks
    summarized_chunks = [summarize_text(chunk) for chunk in chunks]  # Summarize each chunk
    results = []

    for idx, chunk in enumerate(summarized_chunks):
        print(f"Processing chunk {idx + 1}/{len(summarized_chunks)}...")
        result = classify_bias(chunk)
        results.append(result)

    print("\nBias Classification Results:")
    for res in results:
        print(res)


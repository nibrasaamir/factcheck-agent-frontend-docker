import os
import json
from dotenv import load_dotenv
from groq import Groq
from langchain.tools import Tool

# using GROQ_API_KEY 

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# # summarizer model
# _summarizer = pipeline(
#     "summarization",
#     model="sshleifer/distilbart-cnn-12-6",
#     tokenizer="sshleifer/distilbart-cnn-12-6",
#     truncation=True
# )

# # classifier model
# _classifier = pipeline(
#     "zero-shot-classification",
#     model="facebook/bart-large-mnli",
#     tokenizer="facebook/bart-large-mnli"
# )

def summarize_and_classify(text):

    # Groq will summarise the text into 2-3 lines. It will then classify them into support/refure/neutral.
   
    system_prompt = (
        "You are a fact-checking assistant. "
        "Given a passage of text, output ONLY a JSON object with these fields:\n"
        "- summary: a 2–3 sentence summary of the passage\n"
        "- label: one of \"supports\", \"refutes\"\n"
        "- score: a confidence between 0.0 and 1.0\n"
        "Also use your own understanding before deciding if its a support or refute. \n"
        "Do NOT output any additional text."
    )

    user_prompt = f"Passage:\n\"\"\"\n{text}\n\"\"\""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system",  "content": system_prompt},
            {"role": "user",    "content": user_prompt},
        ],
    )
    
    return response.choices[0].message.content.strip()

#---------------------------------------------------------------------------------------------------------

summarizer_tool = Tool(
    name="summarize_and_classify",
    func=summarize_and_classify,
    description=(
        "Use Groq’s LLM to (1) summarize text in 2–3 sentences, "
        "(2) classify the summary as supports/refutes with a confidence score, "
        "and return exactly a JSON object with keys summary, label, and score."
    )
)
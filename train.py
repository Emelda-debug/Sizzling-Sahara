import mwclient
import mwparserfromhell
from openai import OpenAI 
import openai
import os
import pandas as pd
import re
import tiktoken
from dotenv import load_dotenv

load_dotenv()
open_ai_key = os.environ.get("OPEN_AI_KEY")
openai.api_key = open_ai_key


def read_text_document(file_path):
    """
    Read a text document and return a list of strings, where each string is a section of the document.
    """
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    # split the text into sections
    sections = text.split("\n\n")

    # clean up the sections
    sections = [s.strip() for s in sections]
    sections = [s for s in sections if s]

    return sections

file_path = "./ssahara.txt"
sections = read_text_document(file_path)

EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 1000

embeddings = []
for batch_start in range(0, len(sections), BATCH_SIZE):
    batch_end = batch_start + BATCH_SIZE
    batch = sections[batch_start:batch_end]
    print(f"Batch {batch_start} to {batch_end-1}")
    response = openai.embeddings.create(model=EMBEDDING_MODEL, input=batch)
    for i, be in enumerate(response.data):
        assert i == be.index  # double check embeddings are in same order as input
    batch_embeddings = [e.embedding for e in response.data]
    embeddings.extend(batch_embeddings)

df = pd.DataFrame({"text": sections, "embedding": embeddings})
SAVE_PATH = "emdeddings_dataset.csv"

df.to_csv(SAVE_PATH, index=False)

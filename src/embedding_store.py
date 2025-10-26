import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_clean_text(json_path: str) -> str:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)["content"]

def chunk_text(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n", ".", " "]
    )
    chunks = splitter.split_text(text)
    return chunks

def create_embeddings(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)
    return embeddings

def save_embeddings(chunks, embeddings, output_path: str):
    np.savez(output_path, chunks=chunks, embeddings=embeddings)
    print(f"✅ Saved embeddings: {output_path}")

if __name__ == "__main__":
    input_json = Path("data/hr_text.json")
    output_npz = Path("data/hr_embeddings.npz")

    print("Loading clean text...")
    text = load_clean_text(str(input_json))

    print("Chunking text...")
    chunks = chunk_text(text)
    print(f"Total chunks: {len(chunks)}")

    print("Generating embeddings...")
    embeddings = create_embeddings(chunks)

    print("Saving embeddings...")
    save_embeddings(chunks, embeddings, str(output_npz))

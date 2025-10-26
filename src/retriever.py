import numpy as np
import faiss
from pathlib import Path

def build_faiss_index(embeddings: np.ndarray):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def save_index(index, path: str):
    faiss.write_index(index, path)
    print(f"✅ Saved FAISS index to {path}")

def load_index(path: str):
    return faiss.read_index(path)

if __name__ == "__main__":
    data_file = Path("data/hr_embeddings.npz")
    index_file = Path("data/hr_index.faiss")

    if not data_file.exists():
        raise FileNotFoundError(f"{data_file} not found")

    data = np.load(data_file, allow_pickle=True)
    chunks = data["chunks"]
    embeddings = data["embeddings"]

    print("Building FAISS index...")
    index = build_faiss_index(embeddings)

    print("Saving index...")
    save_index(index, str(index_file))

    print(f"Total vectors indexed: {index.ntotal}")

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.preprocessing import MinMaxScaler
import faiss
import json
from pathlib import Path

def load_chunks_and_index():
    data = np.load("data/hr_embeddings.npz", allow_pickle=True)
    chunks = data["chunks"]
    embeddings = data["embeddings"]

    index = faiss.read_index("data/hr_index.faiss")
    return chunks, embeddings, index

def bm25_scores(query, chunks):
    tokenized = [chunk.split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.split())
    return np.array(scores)

def faiss_scores(query_vec, index, k=5):
    query_vec = np.array([query_vec]).astype("float32")
    distances, indices = index.search(query_vec, k)
    return distances[0], indices[0]

def normalize_scores(vec1, vec2):
    scaler = MinMaxScaler()
    merged = np.hstack([vec1.reshape(-1,1), vec2.reshape(-1,1)])
    return scaler.fit_transform(merged)

def hybrid_rerank(query, model, chunks, index, top_k=5):
    query_vec = model.encode([query])[0]
    faiss_dists, faiss_ids = faiss_scores(query_vec, index, k=top_k)
    bm25_sc = bm25_scores(query, chunks)

    norm = normalize_scores(-faiss_dists, bm25_sc[faiss_ids])
    hybrid = 0.6 * norm[:,0] + 0.4 * norm[:,1]
    ranked = sorted(zip(hybrid, faiss_ids), reverse=True)
    results = [{"chunk": chunks[i], "score": float(s)} for s,i in ranked]
    return results

if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    print("Loading data and index...")
    chunks, embeddings, index = load_chunks_and_index()
    model = SentenceTransformer("all-MiniLM-L6-v2")

    query = input("Enter your HR question: ")
    results = hybrid_rerank(query, model, chunks, index)

    print("\nTop relevant chunks:")
    for r in results:
        print(f"\nScore: {r['score']:.3f}\nText: {r['chunk'][:400]}...")

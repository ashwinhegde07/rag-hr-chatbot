from flask import Flask, request, jsonify
from functools import lru_cache
import numpy as np
import faiss
import os, requests
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

app = Flask(__name__)

# ---------- Load data ----------
data = np.load("data/hr_embeddings.npz", allow_pickle=True)
chunks = data["chunks"]
embeddings = data["embeddings"]
index = faiss.read_index("data/hr_index.faiss")
model = SentenceTransformer("all-MiniLM-L6-v2")

tokenized_corpus = [chunk.split() for chunk in chunks]
bm25 = BM25Okapi(tokenized_corpus)

# ---------- Config ----------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_URL = "https://api.groq.com/openai/v1/chat/completions"

# ---------- Utility ----------
def hybrid_search(query, top_k=5):
    q_vec = model.encode([query])[0].astype("float32")
    dists, ids = index.search(np.array([q_vec]), top_k)
    bm25_scores = bm25.get_scores(query.split())
    combined = []
    for dist, idx in zip(dists[0], ids[0]):
        score = 0.6 * (1 / (1 + dist)) + 0.4 * bm25_scores[idx]
        combined.append((score, chunks[idx]))
    combined.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in combined[:top_k]]

@lru_cache(maxsize=64)
def ask_groq(prompt):
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    }
    r = requests.post(LLM_URL, json=payload, headers=headers, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

# ---------- Route ----------
@app.route("/query", methods=["POST"])
def query():
    q = request.json.get("question", "")
    if not q:
        return jsonify({"error": "Missing question"}), 400

    docs = hybrid_search(q)
    context = "\n\n".join(docs)
    prompt = f"You are an HR assistant. Use this context to answer the question accurately.\n\nContext:\n{context}\n\nQuestion: {q}\nAnswer:"

    try:
        answer = ask_groq(prompt)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"answer": answer, "sources": docs})

print("Starting Flask server...")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

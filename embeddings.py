"""
embed_index.py — Embed rulebook chunks with BAAI/bge-small-en-v1.5 and
index into a persistent local ChromaDB collection.

Run AFTER chunking.py.

Run:
    python embed_index.py
"""

import json

import chromadb
from sentence_transformers import SentenceTransformer

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def deco(fn):
            return fn
        return deco

RULEBOOK_DIR = "rulebook"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "chargeback_rulebook"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


@traceable(name="embed_and_index_chunks")
def build_index():
    with open(f"{RULEBOOK_DIR}/chunks.jsonl") as f:
        chunks = [json.loads(line) for line in f]

    print(f"Loaded {len(chunks)} chunks")
    print(f"Loading embedding model: {EMBEDDING_MODEL} (local, ~130MB)")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # BGE models recommend a prefix for passages vs queries
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # fresh collection each run, cosine distance
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=[
            {
                "reason_code": c["reason_code"],
                "section": c["section"],
                "source": c["source"],
            }
            for c in chunks
        ],
    )

    print(f"Indexed {collection.count()} chunks into ChromaDB at {CHROMA_DIR}")
    return collection


if __name__ == "__main__":
    build_index()
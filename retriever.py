"""
retriever.py — Stage 1 retrieval, now HYBRID (keyword + semantic).

Pipeline:
  1. Build a case-aware query from dispute facts.
  2. Filter candidates by reason_code metadata BEFORE any ranking (both
     BM25 and semantic operate only on the reason_code-filtered subset).
  3. Run BM25 keyword search AND semantic (BGE) search independently over
     that filtered subset.
  4. Combine the two rankings via Reciprocal Rank Fusion (RRF) - no score
     normalization needed, RRF only uses rank position, which is why it's
     the standard way to fuse lexical + semantic retrieval.
  5. Return the fused Top-8 for the reranker (reranker.py) to narrow to 3.

Why hybrid: BM25 catches exact-term matches (e.g. "OTP", "tracking",
"duplicate") that a small embedding model can under-weight; semantic search
catches paraphrases/synonyms BM25 misses entirely. At only ~20 chunks the
corpus is small enough that both retrievers are cheap to run every query.

RRF formula (standard, k=60 per the original RRF paper - not tuned further,
it's a well-established default that's robust across corpus sizes):
    RRF_score(doc) = sum over each ranker's rank_list of  1 / (k + rank)

Run standalone for a quick manual check:
    python retriever.py
"""

import re
from collections import defaultdict

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def deco(fn):
            return fn
        return deco

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "chargeback_rulebook"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
TOP_K = 5
RRF_K = 60  # standard RRF constant from the original paper

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str):
    return TOKEN_PATTERN.findall(text.lower())


class RuleRetriever:
    def __init__(self, chroma_dir=CHROMA_DIR, collection_name=COLLECTION_NAME):
        self.client = chromadb.PersistentClient(path=chroma_dir)
        self.collection = self.client.get_collection(collection_name)
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL)

    @traceable(name="build_case_query")
    def build_query(self, reason_code, case_facts: dict):
        parts = [f"Reason code: {reason_code}."]

        evidence_bits = []
        if "delivery_confirmed" in case_facts:
            evidence_bits.append(f"Delivery confirmed: {case_facts['delivery_confirmed']}.")
        if "otp_auth_confirmed" in case_facts:
            evidence_bits.append(f"OTP/authentication confirmed: {case_facts['otp_auth_confirmed']}.")
        if "shipping_billing_match" in case_facts:
            evidence_bits.append(f"Shipping/billing address match: {case_facts['shipping_billing_match']}.")
        if "prior_complaint_on_file" in case_facts:
            evidence_bits.append(f"Prior complaint on file: {case_facts['prior_complaint_on_file']}.")
        if "amount" in case_facts:
            evidence_bits.append(f"Dispute amount: Rs.{case_facts['amount']}.")

        query = " ".join(parts + evidence_bits)
        query += " Determine the evidence required to contest this dispute."
        return query

    def _get_filtered_chunks(self, reason_code):
        """Fetch all chunks for this reason_code once; used to build both
        the BM25 index and as the semantic search candidate pool."""
        result = self.collection.get(
            where={"reason_code": reason_code},
            include=["documents", "metadatas", "embeddings"],
        )
        chunks = []
        for i in range(len(result["ids"])):
            chunks.append({
                "chunk_id": result["ids"][i],
                "text": result["documents"][i],
                "metadata": result["metadatas"][i],
                "embedding": result["embeddings"][i] if result.get("embeddings") is not None else None,
            })
        return chunks

    @traceable(name="bm25_search")
    def _bm25_rank(self, query, chunks):
        """Returns chunk_ids ranked by BM25 score, best first."""
        corpus_tokens = [tokenize(c["text"]) for c in chunks]
        bm25 = BM25Okapi(corpus_tokens)
        scores = bm25.get_scores(tokenize(query))
        ranked_idx = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
        return [chunks[i]["chunk_id"] for i in ranked_idx]

    @traceable(name="semantic_search")
    def _semantic_rank(self, query, chunks):
        """Returns chunk_ids ranked by cosine similarity (via Chroma query
        restricted to this reason_code), best first."""
        query_embedding = self.embed_model.encode(
            BGE_QUERY_PREFIX + query, normalize_embeddings=True
        ).tolist()
        chunk_ids = [c["chunk_id"] for c in chunks]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=len(chunk_ids),
            where={"reason_code": chunks[0]["metadata"]["reason_code"]} if chunks else None,
        )
        return results["ids"][0]

    @staticmethod
    def _reciprocal_rank_fusion(rank_lists, k=RRF_K):
        """rank_lists: list of ordered chunk_id lists (best first), one per
        retriever. Returns chunk_ids sorted by fused RRF score, best first."""
        scores = defaultdict(float)
        for rank_list in rank_lists:
            for rank, chunk_id in enumerate(rank_list, start=1):
                scores[chunk_id] += 1.0 / (k + rank)
        return sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True), dict(scores)

    @traceable(name="retrieve_top_k_chunks_hybrid")
    def retrieve(self, reason_code, case_facts: dict, top_k=TOP_K):
        query = self.build_query(reason_code, case_facts)

        chunks = self._get_filtered_chunks(reason_code)
        n_available = len(chunks)

        if n_available == 0:
            return {
                "query": query,
                "results": [],
                "no_rule_found": True,
            }

        chunk_lookup = {c["chunk_id"]: c for c in chunks}

        bm25_ranked = self._bm25_rank(query, chunks)
        semantic_ranked = self._semantic_rank(query, chunks)

        fused_order, fused_scores = self._reciprocal_rank_fusion([bm25_ranked, semantic_ranked])

        k = min(top_k, n_available)
        candidates = []
        for chunk_id in fused_order[:k]:
            c = chunk_lookup[chunk_id]
            candidates.append({
                "chunk_id": chunk_id,
                "text": c["text"],
                "metadata": c["metadata"],
                "rrf_score": fused_scores[chunk_id],
                "bm25_rank": bm25_ranked.index(chunk_id) + 1,
                "semantic_rank": semantic_ranked.index(chunk_id) + 1,
            })

        return {
            "query": query,
            "results": candidates,
            "no_rule_found": False,
        }


if __name__ == "__main__":
    retriever = RuleRetriever()
    example_facts = {
        "delivery_confirmed": False,
        "otp_auth_confirmed": True,
        "shipping_billing_match": True,
        "prior_complaint_on_file": False,
        "amount": 2400,
    }
    out = retriever.retrieve("item_not_received", example_facts)
    print(f"Query: {out['query']}\n")
    for r in out["results"]:
        print(f"[{r['chunk_id']}] rrf={r['rrf_score']:.4f} "
              f"(bm25_rank={r['bm25_rank']}, semantic_rank={r['semantic_rank']})")
        print(f"  {r['text'][:120]}...\n")
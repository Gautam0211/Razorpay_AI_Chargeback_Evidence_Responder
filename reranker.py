"""
reranker.py — Stage 2: rerank the Top-5 candidates from retriever.py down
to Top-3 using a lightweight local cross-encoder.

Model choice: cross-encoder/ms-marco-TinyBERT-L-2-v2
  - ~17MB, 2-layer TinyBERT distillation — the smallest widely-used
    cross-encoder reranker available via sentence-transformers.
  - Chosen specifically for low RAM footprint on unpaid/local deployment
    (no GPU, no paid inference tier assumed).
  - Trade-off: lower reranking quality ceiling than larger cross-encoders
    (e.g. ms-marco-MiniLM-L-6-v2, ~80MB). Given the corpus is only ~20
    chunks and candidates are already reason_code-filtered before this
    stage, the ranking task is easy enough that TinyBERT's lower capacity
    is an acceptable trade for the RAM savings — verify empirically via
    eval_retriever.py rather than assuming.

Run standalone for a quick manual check:
    python reranker.py
"""

from sentence_transformers import CrossEncoder

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def deco(fn):
            return fn
        return deco

RERANKER_MODEL = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
FINAL_TOP_N = 3


class RuleReranker:
    def __init__(self, model_name=RERANKER_MODEL):
        self.model = CrossEncoder(model_name)

    @traceable(name="rerank_candidates")
    def rerank(self, query, candidates, top_n=FINAL_TOP_N):
        """candidates: list of dicts with at least 'chunk_id' and 'text'
        (as returned by RuleRetriever.retrieve()['results'])."""
        if not candidates:
            return []

        pairs = [[query, c["text"]] for c in candidates]
        scores = self.model.predict(pairs)

        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)

        ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
        return ranked[:top_n]


if __name__ == "__main__":
    from retriever import RuleRetriever

    retriever = RuleRetriever()
    reranker = RuleReranker()

    example_facts = {
        "delivery_confirmed": False,
        "otp_auth_confirmed": True,
        "shipping_billing_match": True,
        "prior_complaint_on_file": False,
        "amount": 2400,
    }
    retrieval = retriever.retrieve("item_not_received", example_facts)
    top3 = reranker.rerank(retrieval["query"], retrieval["results"])

    print(f"Retrieved {len(retrieval['results'])} candidates, reranked to {len(top3)}\n")
    for r in top3:
        print(f"[{r['chunk_id']}] rerank_score={r['rerank_score']:.4f} (orig rrf_score={r['rrf_score']:.4f})")
        print(f"  {r['text'][:120]}...\n")
"""
eval_retriever.py — Evaluates retriever.py and reranker.py SEPARATELY from
generation, using golden_answers.csv's correct_relevant_chunk_id.

Granularity note: golden_answers.csv labels the correct RULE (per reason
code), not a specific section-level chunk (chunking.py splits each rule
into ~5 section chunks). So "correct" here means: a returned chunk's
reason_code metadata matches the dispute's true reason_code. This is the
right granularity for this golden set — section-level relevance would need
new golden labels we don't have, and rule-level is what the golden data was
actually designed to test (see generate_data.py's CHUNK_ID_MAP).

Also evaluates the unknown-reason edge cases (other_unclassified): correct
behavior is retrieving ZERO candidates (routes to manual review), not a
low-scoring match.

Metrics:
  Stage 1 (Chroma, before rerank): Recall@5
    - did >=1 chunk with the correct reason_code appear in the Top-5?
  Stage 2 (after reranker): Precision@3, Recall@3
    - Precision@3: of the Top-3 final chunks, what fraction have the
      correct reason_code?
    - Recall@3: did >=1 correct-reason_code chunk survive into Top-3?

Run:
    python eval_retriever.py
"""

import json

import pandas as pd

from retriever import RuleRetriever
from reranker import RuleReranker

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def deco(fn):
            return fn
        return deco

DATA_DIR = "data"  # adjust if running from a different cwd
N_SAMPLE_PER_REASON = 30  # keep eval fast; full corpus is only 4 reason codes anyway


def sample_eval_set():
    disputes = pd.read_csv(f"{DATA_DIR}/disputes.csv")
    golden = pd.read_csv(f"{DATA_DIR}/golden_answers.csv")
    orders = pd.read_csv(f"{DATA_DIR}/orders.csv")

    df = disputes.merge(golden, on="dispute_id").merge(orders, on="order_id")

    samples = []
    for reason_code, group in df.groupby("reason_code"):
        samples.append(group.sample(min(N_SAMPLE_PER_REASON, len(group)), random_state=42))
    return pd.concat(samples).reset_index(drop=True)


def case_facts_from_row(row):
    return {
        "amount": row["amount"],
        "delivery_confirmed": row["delivery_confirmed"],
        "otp_auth_confirmed": row["otp_auth_confirmed"],
        "shipping_billing_match": row["shipping_billing_match"],
        "prior_complaint_on_file": row["prior_complaint_on_file"],
    }


@traceable(name="evaluate_retriever_and_reranker")
def run_eval():
    retriever = RuleRetriever()
    reranker = RuleReranker()

    eval_df = sample_eval_set()
    print(f"Evaluating on {len(eval_df)} sampled disputes across reason codes\n")

    recall_at_5_hits = 0
    recall_at_3_hits = 0
    precision_at_3_sum = 0.0
    n_with_rule = 0

    unknown_correct = 0
    n_unknown = 0

    for _, row in eval_df.iterrows():
        reason_code = row["reason_code"]
        true_reason = reason_code  # golden granularity == reason_code, see module docstring
        facts = case_facts_from_row(row)

        retrieval = retriever.retrieve(reason_code, facts)

        if reason_code == "other_unclassified":
            n_unknown += 1
            if retrieval["no_rule_found"]:
                unknown_correct += 1
            continue

        n_with_rule += 1
        top5 = retrieval["results"]

        hit_at_8 = any(c["metadata"]["reason_code"] == true_reason for c in top5)
        recall_at_5_hits += int(hit_at_8)

        top3 = reranker.rerank(retrieval["query"], top5, top_n=3)
        correct_in_top3 = sum(1 for c in top3 if c["metadata"]["reason_code"] == true_reason)

        precision_at_3_sum += correct_in_top3 / max(len(top3), 1)
        recall_at_3_hits += int(correct_in_top3 > 0)

    print("=" * 60)
    print("RETRIEVER + RERANKER EVALUATION")
    print("=" * 60)
    if n_with_rule:
        print(f"Recall@8   : {recall_at_5_hits / n_with_rule:.3f}  ({recall_at_5_hits}/{n_with_rule})")
        print(f"Recall@3   : {recall_at_3_hits / n_with_rule:.3f}  ({recall_at_3_hits}/{n_with_rule})")
        print(f"Precision@3: {precision_at_3_sum / n_with_rule:.3f}")
    if n_unknown:
        print(f"\nUnknown-reason correct fallback (no_rule_found=True): "
              f"{unknown_correct}/{n_unknown} ({unknown_correct/n_unknown:.1%})")

    results = {
        "n_with_rule": n_with_rule,
        "recall_at_8": recall_at_5_hits / n_with_rule if n_with_rule else None,
        "recall_at_3": recall_at_3_hits / n_with_rule if n_with_rule else None,
        "precision_at_3": precision_at_3_sum / n_with_rule if n_with_rule else None,
        "n_unknown": n_unknown,
        "unknown_correct_fallback_rate": unknown_correct / n_unknown if n_unknown else None,
    }
    with open("retriever_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to retriever_eval_results.json")
    return results


if __name__ == "__main__":
    run_eval()
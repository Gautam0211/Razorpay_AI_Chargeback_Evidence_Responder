"""
pipeline.py — Plain (non-LangGraph) end-to-end integration.

Connects the components already built and tested individually:

    XGBoost scorer
        ↓
    Decision + Priority layer
        ↓
    Hybrid RAG retriever
        ↓
    Cross-encoder reranker
        ↓
    LLM evidence generator
        ↓
    Validator

Entry point:
    result = process_dispute(dispute_id)

This is intentionally plain Python.
LangGraph will be added later as the orchestration/controller layer.

Run:
    python pipeline.py

The standalone test checks:
    1. Strong FIGHT
    2. Strong CONCEDE
    3. MANUAL_REVIEW
    4. other_unclassified fallback
"""

import json
import os
from pathlib import Path

import joblib
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"


# ============================================================
# COMPONENT IMPORTS
# ============================================================

from retriever import RuleRetriever
from reranker import RuleReranker
from generator import EvidenceGenerator
from validator import validate


# ============================================================
# LANGSMITH
# ============================================================

try:
    from langsmith import traceable
except ImportError:

    def traceable(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


# ============================================================
# DECISION-LAYER CONFIGURATION
# Must match the values already selected during validation.
# ============================================================

FIGHT_COST_FLAT = 300.0
REVIEW_MARGIN = 150.0

MAX_RESPONSE_WINDOW_DAYS = 20

AMOUNT_WEIGHT = 0.6
URGENCY_WEIGHT = 0.4

MAX_DRAFT_RETRIES = 2


# ============================================================
# PIPELINE
# ============================================================

class DisputePipeline:

    def __init__(self):

        print("Loading pipeline components...")

        # ----------------------------------------------------
        # ML model
        # ----------------------------------------------------

        model_path = MODEL_DIR / "xgb_scorer.joblib"
        feature_path = MODEL_DIR / "feature_columns.json"

        if not model_path.exists():
            raise FileNotFoundError(
                f"ML model not found: {model_path}"
            )

        if not feature_path.exists():
            raise FileNotFoundError(
                f"Feature columns file not found: {feature_path}"
            )

        self.model = joblib.load(model_path)

        with open(feature_path, "r", encoding="utf-8") as f:
            self.feature_columns = json.load(f)

        # ----------------------------------------------------
        # Data
        # ----------------------------------------------------

        orders_path = DATA_DIR / "orders.csv"
        disputes_path = DATA_DIR / "disputes.csv"

        if not orders_path.exists():
            raise FileNotFoundError(
                f"Orders file not found: {orders_path}"
            )

        if not disputes_path.exists():
            raise FileNotFoundError(
                f"Disputes file not found: {disputes_path}"
            )

        self.orders = pd.read_csv(orders_path)
        self.disputes = pd.read_csv(disputes_path)

        # ----------------------------------------------------
        # Fixed amount range for comparable priority scores
        # across separate pipeline calls.
        # ----------------------------------------------------

        joined = self.disputes.merge(
            self.orders,
            on="order_id",
            how="inner"
        )

        if joined.empty:
            raise ValueError(
                "Could not join disputes.csv with orders.csv."
            )

        self._amount_min = float(joined["amount"].min())
        self._amount_max = float(joined["amount"].max())

        # ----------------------------------------------------
        # RAG
        # ----------------------------------------------------

        self.retriever = RuleRetriever()
        self.reranker = RuleReranker()

        # ----------------------------------------------------
        # LLM
        # ----------------------------------------------------

        self.generator = EvidenceGenerator()

        print("Pipeline ready.\n")


    # ========================================================
    # 1. LOAD CASE
    # ========================================================

    @traceable(name="load_case_facts")
    def _load_case_facts(self, dispute_id):

        dispute_rows = self.disputes[
            self.disputes["dispute_id"] == dispute_id
        ]

        if dispute_rows.empty:
            raise ValueError(
                f"Unknown dispute_id: {dispute_id}"
            )

        dispute_row = dispute_rows.iloc[0]

        order_rows = self.orders[
            self.orders["order_id"] == dispute_row["order_id"]
        ]

        if order_rows.empty:
            raise ValueError(
                f"No matching order for dispute {dispute_id}"
            )

        order_row = order_rows.iloc[0]

        case_facts = {
            "amount": float(order_row["amount"]),

            "item_category": str(
                order_row["item_category"]
            ),

            "delivery_confirmed": bool(
                order_row["delivery_confirmed"]
            ),

            "otp_auth_confirmed": bool(
                order_row["otp_auth_confirmed"]
            ),

            "shipping_billing_match": bool(
                order_row["shipping_billing_match"]
            ),

            "prior_complaint_on_file": bool(
                order_row["prior_complaint_on_file"]
            ),

            "days_to_dispute": int(
                dispute_row["days_to_dispute"]
            ),

            # Current dataset/pipeline does not track this live.
            # Keep it consistent with the current implementation.
            "prior_dispute_count_at_time": 0,
        }

        reason_code = str(dispute_row["reason_code"])

        return case_facts, reason_code


    # ========================================================
    # 2. ML SCORING
    # ========================================================

    @traceable(name="score_dispute")
    def _score(self, case_facts, reason_code):

        row = pd.DataFrame([
            {
                **case_facts,
                "reason_code": reason_code
            }
        ])

        # ----------------------------------------------------
        # Numeric features
        # ----------------------------------------------------

        X = row[
            [
                "amount",
                "days_to_dispute",
                "prior_dispute_count_at_time",
            ]
        ].copy()

        # ----------------------------------------------------
        # Boolean features
        # ----------------------------------------------------

        boolean_features = [
            "delivery_confirmed",
            "otp_auth_confirmed",
            "shipping_billing_match",
            "prior_complaint_on_file",
        ]

        for col in boolean_features:
            X[col] = row[col].astype(int)

        # ----------------------------------------------------
        # Categorical features
        # ----------------------------------------------------

        categorical_features = [
            "reason_code",
            "item_category",
        ]

        cat_dummies = pd.get_dummies(
            row[categorical_features],
            prefix=categorical_features
        )

        X = pd.concat(
            [X, cat_dummies],
            axis=1
        )

        # ----------------------------------------------------
        # Align exactly with training feature columns
        # ----------------------------------------------------

        for col in self.feature_columns:

            if col not in X.columns:
                X[col] = 0

        X = X[self.feature_columns]

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        win_probability = float(
            self.model.predict_proba(X)[:, 1][0]
        )

        return win_probability


    # ========================================================
    # 3. DECISION + PRIORITY
    # ========================================================

    @traceable(name="decide_and_prioritize")
    def _decide(
        self,
        win_probability,
        amount,
        days_to_dispute
    ):

        # ----------------------------------------------------
        # Expected value of fighting
        # ----------------------------------------------------

        ev_fight = (
            win_probability * amount
            - FIGHT_COST_FLAT
        )

        # ----------------------------------------------------
        # Decision
        # ----------------------------------------------------

        if ev_fight > REVIEW_MARGIN:

            decision = "FIGHT"

        elif ev_fight < -REVIEW_MARGIN:

            decision = "CONCEDE"

        else:

            decision = "MANUAL_REVIEW"

        # ----------------------------------------------------
        # Urgency
        # ----------------------------------------------------

        days_left = max(
            MAX_RESPONSE_WINDOW_DAYS - days_to_dispute,
            0
        )

        # ----------------------------------------------------
        # Amount normalization
        # ----------------------------------------------------

        amount_norm = (
            (amount - self._amount_min)
            /
            (
                self._amount_max
                - self._amount_min
                + 1e-9
            )
        )

        amount_norm = max(
            0.0,
            min(1.0, amount_norm)
        )

        # ----------------------------------------------------
        # Urgency score
        # ----------------------------------------------------

        urgency = 1 - min(
            days_left / MAX_RESPONSE_WINDOW_DAYS,
            1
        )

        # ----------------------------------------------------
        # Priority
        # ----------------------------------------------------

        priority_score = (
            AMOUNT_WEIGHT * amount_norm
            +
            URGENCY_WEIGHT * urgency
        )

        return {
            "decision": decision,
            "ev_fight": round(ev_fight, 2),
            "days_left": days_left,
            "priority_score": round(
                priority_score,
                4
            ),
        }


    # ========================================================
    # 4. PROCESS AN ALREADY-CONSTRUCTED CASE
    # ========================================================
    #
    # This is the ONE processing path in the pipeline.
    # process_dispute() is just a CSV-lookup wrapper around
    # this method. The Manual Analysis UI, and the live
    # simulation path, both call this directly with a
    # hand-built / registered case_facts dict. No second
    # scoring/decision/RAG implementation exists anywhere.
    # ========================================================

    @traceable(name="process_case")
    def process_case(self, dispute_id, case_facts, reason_code):

        # ----------------------------------------------------
        # ML scoring
        # ----------------------------------------------------

        win_probability = self._score(
            case_facts,
            reason_code
        )

        # ----------------------------------------------------
        # Decision + priority
        # ----------------------------------------------------

        decision_info = self._decide(
            win_probability,
            case_facts["amount"],
            case_facts["days_to_dispute"]
        )

        # ----------------------------------------------------
        # Base result
        # ----------------------------------------------------

        result = {
            "dispute_id": dispute_id,

            "reason_code": reason_code,

            "case_facts": case_facts,

            "win_probability": round(
                win_probability,
                4
            ),

            **decision_info,

            "retrieved_chunks": None,

            "draft": None,

            "draft_status": None,

            "validation": None,
        }

        # ====================================================
        # UNKNOWN REASON FALLBACK
        # ====================================================

        if reason_code == "other_unclassified":

            result["decision"] = "MANUAL_REVIEW"

            result["draft_status"] = "NO_RULE_FOUND"

            result["note"] = (
                "Unrecognized reason code - "
                "no rulebook entry. "
                "Routed to manual review."
            )

            return result

        # ====================================================
        # RAG RETRIEVAL
        # ====================================================

        retrieval = self.retriever.retrieve(
            reason_code,
            case_facts
        )

        # ----------------------------------------------------
        # No applicable rule
        # ----------------------------------------------------

        if retrieval["no_rule_found"]:

            result["decision"] = "MANUAL_REVIEW"

            result["draft_status"] = "NO_RULE_FOUND"

            result["note"] = (
                "No applicable rule retrieved. "
                "Routed to manual review."
            )

            return result

        # ====================================================
        # RERANK
        # ====================================================

        top3 = self.reranker.rerank(
            retrieval["query"],
            retrieval["results"]
        )

        result["retrieved_chunks"] = [
            {
                "chunk_id": c["chunk_id"],
                "section": c["metadata"]["section"],
                "rerank_score": round(
                    float(c["rerank_score"]),
                    4
                ),
            }
            for c in top3
        ]

        # ====================================================
        # CONCEDE
        # ====================================================

        if decision_info["decision"] == "CONCEDE":

            result["draft_status"] = (
                "SKIPPED_CONCEDE"
            )

            result["note"] = (
                "EV favors concede; "
                "no evidence draft generated."
            )

            return result

        # ====================================================
        # FIGHT / MANUAL REVIEW
        # ====================================================

        draft_result = self._generate_with_retry(
            dispute_id=dispute_id,
            reason_code=reason_code,
            case_facts=case_facts,
            top3=top3,
            win_probability=win_probability,
            decision=decision_info["decision"],
        )

        result.update(draft_result)

        return result


    # ========================================================
    # 5. LOAD-FROM-CSV WRAPPER
    # ========================================================
    #
    # Loads a case from the stored disputes/orders CSVs, then
    # runs it through the shared process_case() logic. This is
    # the path used for the 15,000-case historical queue.
    # ========================================================

    @traceable(name="process_dispute")
    def process_dispute(self, dispute_id):

        case_facts, reason_code = self._load_case_facts(dispute_id)
        return self.process_case(dispute_id, case_facts, reason_code)


    # ========================================================
    # 6. LLM GENERATION + VALIDATION + RETRY
    # ========================================================

    @traceable(name="generate_with_retry")
    def _generate_with_retry(
        self,
        dispute_id,
        reason_code,
        case_facts,
        top3,
        win_probability,
        decision,
    ):

        attempt = 0

        last_validation = None

        while attempt <= MAX_DRAFT_RETRIES:

            # ------------------------------------------------
            # Generate
            # ------------------------------------------------

            gen = self.generator.generate(
                dispute_id=dispute_id,
                reason_code=reason_code,
                case_facts=case_facts,
                retrieved_chunks=top3,
                win_probability=win_probability,
                decision=decision,
            )

            # ------------------------------------------------
            # Generator failure
            # ------------------------------------------------

            if gen["status"] != "DRAFTED":

                return {
                    "draft": None,

                    "draft_status": gen["status"],

                    "validation": None,

                    "note": gen.get(
                        "note",
                        "LLM generation failed."
                    ),
                }

            # ------------------------------------------------
            # Validate generated draft
            # ------------------------------------------------

            validation_result = validate(
                gen["draft"],
                dispute_id,
                case_facts,
                top3
            )

            last_validation = validation_result

            # ------------------------------------------------
            # Successful validation
            # ------------------------------------------------

            if validation_result["passed"]:

                return {
                    "draft": gen["draft"],

                    "draft_status": "VALIDATED",

                    "validation": validation_result,
                }

            # ------------------------------------------------
            # Validation failed → retry
            # ------------------------------------------------

            attempt += 1

        # ====================================================
        # MAX RETRIES EXHAUSTED
        # ====================================================

        return {
            "draft": None,

            "draft_status": (
                "VALIDATION_FAILED_MAX_RETRIES"
            ),

            "validation": last_validation,

            "note": (
                "Draft failed grounding/field "
                "validation after retries. "
                "Routed to manual review."
            ),
        }


    # ========================================================
    # 7. LIVE SIMULATION SUPPORT
    # In-memory only — does not touch the historical CSVs on
    # disk. Lets newly "arrived" disputes flow through the
    # exact same process_dispute() path as historical cases.
    # ========================================================

    def register_live_cases(self, incoming_df):
        """
        Appends newly 'arrived' disputes/orders into this pipeline
        instance's in-memory tables so process_dispute(dispute_id)
        works for them exactly like historical cases.

        incoming_df columns expected:
            dispute_id, order_id, reason_code, amount, item_category,
            delivery_confirmed, otp_auth_confirmed,
            shipping_billing_match, prior_complaint_on_file,
            days_to_dispute
        """
        order_cols = [
            "order_id", "amount", "item_category",
            "delivery_confirmed", "otp_auth_confirmed",
            "shipping_billing_match", "prior_complaint_on_file",
        ]
        dispute_cols = [
            "dispute_id", "order_id", "reason_code", "days_to_dispute",
        ]

        new_orders = incoming_df[order_cols].copy()
        new_disputes = incoming_df[dispute_cols].copy()
        new_disputes["status"] = "pending"

        # Avoid duplicate registration across reruns.
        existing_order_ids = set(self.orders["order_id"])
        existing_dispute_ids = set(self.disputes["dispute_id"])

        new_orders = new_orders[~new_orders["order_id"].isin(existing_order_ids)]
        new_disputes = new_disputes[~new_disputes["dispute_id"].isin(existing_dispute_ids)]

        if not new_orders.empty:
            self.orders = pd.concat([self.orders, new_orders], ignore_index=True)
        if not new_disputes.empty:
            self.disputes = pd.concat([self.disputes, new_disputes], ignore_index=True)


# ============================================================
# MODULE-LEVEL CONVENIENCE FUNCTION
# ============================================================

_pipeline_cache = None


def process_dispute(dispute_id):
    """
    Convenience wrapper.

    Example:
        result = process_dispute("DISPUTE_ID")
    """

    global _pipeline_cache

    if _pipeline_cache is None:
        _pipeline_cache = DisputePipeline()

    return _pipeline_cache.process_dispute(
        dispute_id
    )


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("AI CHARGEBACK EVIDENCE RESPONDER")
    print("End-to-End Pipeline Test")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    disputes_path = DATA_DIR / "disputes.csv"
    golden_path = DATA_DIR / "golden_answers.csv"

    disputes = pd.read_csv(disputes_path)
    golden = pd.read_csv(golden_path)

    joined = disputes.merge(
        golden,
        on="dispute_id",
        how="inner"
    )

    # --------------------------------------------------------
    # Select test cases
    # --------------------------------------------------------

    eligible = joined[
        ~joined["is_unknown_reason_edge_case"]
    ]

    # Strong FIGHT candidate
    strong_fight_rows = eligible[
        eligible["true_win_probability"] > 0.75
    ]

    # Strong CONCEDE candidate
    strong_concede_rows = eligible[
        eligible["true_win_probability"] < 0.15
    ]

    # Manual review candidate
    manual_review_rows = eligible[
        eligible["true_win_probability"].between(
            0.35,
            0.55
        )
    ]

    # Unknown reason
    unknown_rows = joined[
        joined["is_unknown_reason_edge_case"]
    ]

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    if strong_fight_rows.empty:
        raise RuntimeError(
            "Could not find a strong FIGHT test case."
        )

    if strong_concede_rows.empty:
        raise RuntimeError(
            "Could not find a strong CONCEDE test case."
        )

    if manual_review_rows.empty:
        raise RuntimeError(
            "Could not find a MANUAL_REVIEW test case."
        )

    if unknown_rows.empty:
        raise RuntimeError(
            "Could not find an unknown-reason test case."
        )

    # --------------------------------------------------------
    # Build test-case dictionary
    # --------------------------------------------------------

    test_cases = {

        "STRONG FIGHT":
            strong_fight_rows.iloc[0]["dispute_id"],

        "STRONG CONCEDE":
            strong_concede_rows.iloc[0]["dispute_id"],

        "MANUAL_REVIEW CANDIDATE":
            manual_review_rows.iloc[0]["dispute_id"],

        "UNKNOWN REASON FALLBACK":
            unknown_rows.iloc[0]["dispute_id"],
    }

    # --------------------------------------------------------
    # Initialize pipeline once
    # --------------------------------------------------------

    pipeline = DisputePipeline()

    # --------------------------------------------------------
    # Run cases
    # --------------------------------------------------------

    for label, dispute_id in test_cases.items():

        print()
        print("=" * 70)
        print(label)
        print(f"Dispute ID: {dispute_id}")
        print("=" * 70)

        try:

            result = pipeline.process_dispute(
                dispute_id
            )

            # ------------------------------------------------
            # Print everything except long LLM draft
            # ------------------------------------------------

            summary = {
                key: value
                for key, value in result.items()
                if key != "draft"
            }

            print(
                json.dumps(
                    summary,
                    indent=2,
                    default=str
                )
            )

            # ------------------------------------------------
            # Print draft separately
            # ------------------------------------------------

            if result.get("draft"):

                print()
                print("--- GENERATED DRAFT ---")
                print()
                print(result["draft"])

        except Exception as e:

            print()
            print("PIPELINE ERROR")
            print(type(e).__name__)
            print(str(e))

        print()
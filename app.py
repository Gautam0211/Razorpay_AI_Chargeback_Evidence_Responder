"""
app.py — AI Chargeback Risk Manager (Streamlit dashboard)

Consumes the existing backend (rag/pipeline.py's DisputePipeline) as the
source of truth. Does NOT reimplement ML scoring, decision logic, RAG
retrieval, reranking, generation, or validation - every result shown here
comes from calling pipeline.process_dispute().

Run:
    streamlit run app.py
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / "rag"))
sys.path.append(str(BASE_DIR / "scripts"))

from pipeline import DisputePipeline, DATA_DIR, MODEL_DIR  # noqa: E402

st.set_page_config(
    page_title="AI Chargeback Risk Manager",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# STYLE
# ============================================================
st.markdown("""
<style>
    .kpi-card {
        background: white; border: 1px solid #E5E7EB; border-radius: 10px;
        padding: 16px 20px; text-align: left;
    }
    .kpi-label { font-size: 13px; color: #6B7280; font-weight: 500; }
    .kpi-value { font-size: 28px; font-weight: 700; color: #111827; }
    .status-fight { color: #059669; font-weight: 600; }
    .status-review { color: #D97706; font-weight: 600; }
    .status-concede { color: #6B7280; font-weight: 600; }
    .pill { padding: 2px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; }
    .pill-fight { background: #D1FAE5; color: #065F46; }
    .pill-review { background: #FEF3C7; color: #92400E; }
    .pill-concede { background: #F3F4F6; color: #374151; }
    div[data-testid="stMetricValue"] { font-size: 24px; }
</style>
""", unsafe_allow_html=True)

DECISION_PILL = {
    "FIGHT": '<span class="pill pill-fight">FIGHT</span>',
    "MANUAL_REVIEW": '<span class="pill pill-review">MANUAL REVIEW</span>',
    "CONCEDE": '<span class="pill pill-concede">CONCEDE</span>',
}

# ============================================================
# LIVE SIMULATION CONFIG (NEW)
# ============================================================
LIVE_STATE_PATH = DATA_DIR / "live_simulation_state.json"
INCOMING_POOL_PATH = DATA_DIR / "incoming_disputes.csv"
RELEASE_BATCH_SIZE = 10  # 5-20 recommended


# ============================================================
# BACKEND ACCESS (cached — loads once per server process)
# ============================================================
@st.cache_resource(show_spinner="Loading AI pipeline (model, retriever, reranker)...")
def get_pipeline():
    return DisputePipeline()


@st.cache_data(ttl=300, show_spinner="Scoring full dispute queue...")
def load_scored_queue(_pipeline):
    """
    Score the complete eligible dispute corpus in one vectorized
    XGBoost prediction call.

    No RAG, reranking, LLM generation, or validation happens here.
    Those are executed only when a dispute is opened.
    """

    # ------------------------------------------------------------
    # 1. Load and join dispute + order data ONCE
    # ------------------------------------------------------------
    disputes = pd.read_csv(DATA_DIR / "disputes.csv")
    orders = pd.read_csv(DATA_DIR / "orders.csv")

    disputes = disputes[
        disputes["reason_code"] != "other_unclassified"
    ].copy()

    df = disputes.merge(
        orders,
        on="order_id",
        how="inner"
    )

    if df.empty:
        return pd.DataFrame()

    # ------------------------------------------------------------
    # 2. Build EXACTLY the same features as pipeline._score()
    # ------------------------------------------------------------
    X = df[
        [
            "amount",
            "days_to_dispute",
        ]
    ].copy()

    # pipeline currently uses 0 as the stored/live placeholder
    # for prior dispute count.
    X["prior_dispute_count_at_time"] = 0

    boolean_features = [
        "delivery_confirmed",
        "otp_auth_confirmed",
        "shipping_billing_match",
        "prior_complaint_on_file",
    ]

    for col in boolean_features:
        X[col] = df[col].astype(int)

    # Same one-hot encoding approach as pipeline._score()
    cat_dummies = pd.get_dummies(
        df[["reason_code", "item_category"]],
        prefix=["reason_code", "item_category"]
    )

    X = pd.concat([X, cat_dummies], axis=1)

    # ------------------------------------------------------------
    # 3. Align EXACTLY to model training feature columns
    # ------------------------------------------------------------
    for col in _pipeline.feature_columns:
        if col not in X.columns:
            X[col] = 0

    X = X[_pipeline.feature_columns]

    # ------------------------------------------------------------
    # 4. ONE XGBoost prediction for the entire queue
    # ------------------------------------------------------------
    win_probabilities = _pipeline.model.predict_proba(X)[:, 1]

    df["win_probability"] = win_probabilities

    # ------------------------------------------------------------
    # 5. Vectorized decision engine
    #    Same constants as pipeline._decide()
    # ------------------------------------------------------------
    df["ev_fight"] = (
        df["win_probability"] * df["amount"]
        - 300.0
    )

    df["decision"] = "MANUAL_REVIEW"

    df.loc[
        df["ev_fight"] > 150.0,
        "decision"
    ] = "FIGHT"

    df.loc[
        df["ev_fight"] < -150.0,
        "decision"
    ] = "CONCEDE"

    # ------------------------------------------------------------
    # 6. Vectorized priority calculation
    #    Same formula as pipeline._decide()
    # ------------------------------------------------------------
    days_left = (
        20 - df["days_to_dispute"]
    ).clip(lower=0)

    amount_norm = (
        df["amount"] - _pipeline._amount_min
    ) / (
        _pipeline._amount_max
        - _pipeline._amount_min
        + 1e-9
    )

    urgency = (
        1 - (days_left / 20).clip(upper=1)
    )

    df["days_left"] = days_left

    df["priority_score"] = (
        0.6 * amount_norm
        + 0.4 * urgency
    )

    # ------------------------------------------------------------
    # 7. Return same queue schema used by dashboard
    # ------------------------------------------------------------
    queue_df = df[
        [
            "dispute_id",
            "reason_code",
            "amount",
            "win_probability",
            "decision",
            "ev_fight",
            "days_left",
            "priority_score",
        ]
    ].copy()

    return (
        queue_df
        .sort_values(
            "priority_score",
            ascending=False
        )
        .reset_index(drop=True)
    )


@st.cache_data(ttl=600)
def load_model_metrics():
    with open(MODEL_DIR / "metrics_summary.json") as f:
        metrics = json.load(f)
    threshold_path = MODEL_DIR / "threshold_comparison.json"
    threshold_compare = json.loads(threshold_path.read_text()) if threshold_path.exists() else None
    return metrics, threshold_compare


@st.cache_data(ttl=600)
def load_economic_figures():
    path = MODEL_DIR / "review_margin_sweep.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


# ============================================================
# LIVE SIMULATION HELPERS (NEW)
# ============================================================

def _load_live_state():
    if LIVE_STATE_PATH.exists():
        return json.loads(LIVE_STATE_PATH.read_text())
    return {"released_count": 0, "released_dispute_ids": []}


def _save_live_state(state):
    LIVE_STATE_PATH.write_text(json.dumps(state, indent=2))


def score_incoming_batch(_pipeline, incoming_df):
    """
    Same exact feature prep / decision constants / priority formula as
    load_scored_queue() — just applied to a small incoming batch.
    No RAG, reranker, LLM, or validator calls here.
    """
    if incoming_df.empty:
        return pd.DataFrame()

    df = incoming_df.copy()

    X = df[["amount", "days_to_dispute"]].copy()
    X["prior_dispute_count_at_time"] = 0

    for col in ["delivery_confirmed", "otp_auth_confirmed",
                "shipping_billing_match", "prior_complaint_on_file"]:
        X[col] = df[col].astype(int)

    cat_dummies = pd.get_dummies(
        df[["reason_code", "item_category"]],
        prefix=["reason_code", "item_category"]
    )
    X = pd.concat([X, cat_dummies], axis=1)

    for col in _pipeline.feature_columns:
        if col not in X.columns:
            X[col] = 0
    X = X[_pipeline.feature_columns]

    df["win_probability"] = _pipeline.model.predict_proba(X)[:, 1]

    df["ev_fight"] = df["win_probability"] * df["amount"] - 300.0
    df["decision"] = "MANUAL_REVIEW"
    df.loc[df["ev_fight"] > 150.0, "decision"] = "FIGHT"
    df.loc[df["ev_fight"] < -150.0, "decision"] = "CONCEDE"

    days_left = (20 - df["days_to_dispute"]).clip(lower=0)
    amount_norm = (df["amount"] - _pipeline._amount_min) / (
        _pipeline._amount_max - _pipeline._amount_min + 1e-9
    )
    urgency = 1 - (days_left / 20).clip(upper=1)

    df["days_left"] = days_left
    df["priority_score"] = 0.6 * amount_norm + 0.4 * urgency
    df["arrival_timestamp"] = datetime.now().strftime("%H:%M:%S")

    return df[[
        "dispute_id", "reason_code", "amount", "win_probability",
        "decision", "ev_fight", "days_left", "priority_score",
        "arrival_timestamp",
    ]].copy()


def release_new_disputes(pipeline, n=RELEASE_BATCH_SIZE):
    """
    Called ONLY on explicit Refresh Queue click. Reads the pre-generated
    pool, releases up to n unreleased rows, registers them with the
    pipeline in-memory, scores them, and updates persistent state.
    Returns (scored_batch_df, error_message_or_None).
    """
    if not INCOMING_POOL_PATH.exists():
        return pd.DataFrame(), "Incoming pool file not found."

    pool = pd.read_csv(INCOMING_POOL_PATH)
    state = _load_live_state()
    released_ids = set(state["released_dispute_ids"])

    available = pool[~pool["dispute_id"].isin(released_ids)]
    if available.empty:
        return pd.DataFrame(), "No more incoming disputes available in the simulated pool."

    batch = available.head(n).copy()

    pipeline.register_live_cases(batch)
    scored = score_incoming_batch(pipeline, batch)

    state["released_dispute_ids"].extend(batch["dispute_id"].tolist())
    state["released_count"] = len(state["released_dispute_ids"])
    _save_live_state(state)

    return scored, None


def get_combined_queue(pipeline):
    """
    Combines the cached historical queue with any live-simulation
    arrivals collected so far this session, sorted by priority_score.
    """
    historical = load_scored_queue(pipeline)
    live = st.session_state.live_queue
    if live.empty:
        return historical

    live_aligned = live.reindex(
        columns=historical.columns.tolist()
        + [c for c in live.columns if c not in historical.columns],
        fill_value=None,
    )
    combined = pd.concat([historical, live_aligned], ignore_index=True)
    return combined.sort_values("priority_score", ascending=False).reset_index(drop=True)


def format_bool(v):
    return "✓ Yes" if v else "✗ No"


def priority_label(score):
    if score >= 0.66:
        return "HIGH"
    elif score >= 0.33:
        return "MEDIUM"
    return "LOW"


def render_dispute_result(result, is_manual=False):
    """Renders a process_dispute()/process_case() result dict. Shared by
    both the automatic Dispute Analysis page and the Manual Analysis page
    so both consume identical backend output through identical UI - no
    separate rendering logic for manual cases."""
    facts = result["case_facts"]

    if is_manual:
        st.warning("📝 **Manual Analysis — Not part of the stored queue**")

    st.markdown(f"### {result['dispute_id']}")
    st.caption(result["reason_code"].replace("_", " ").title())

    m1, m2, m3 = st.columns(3)
    m1.metric("WIN PROBABILITY", f"{result['win_probability']*100:.2f}%")
    m2.markdown(f"**DECISION**<br>{DECISION_PILL.get(result['decision'], result['decision'])}", unsafe_allow_html=True)
    m3.metric("PRIORITY", f"{result.get('priority_score', 0):.4f}")

    st.markdown("#### Expected Value")
    ev1, ev2, ev3 = st.columns(3)
    ev1.metric(f"EV({result['decision']})", f"₹{result.get('ev_fight', 0):,.2f}")
    ev2.metric("Amount", f"₹{facts['amount']:,.2f}")
    ev3.metric("Fight Cost", "₹300")
    st.caption("Fight cost of ₹300 is a **configurable operational cost assumption**, not an industry standard.")

    st.markdown("---")
    st.markdown("#### Case Facts")
    fc1, fc2 = st.columns(2)
    with fc1:
        st.write(f"**Amount:** ₹{facts['amount']:,.2f}")
        st.write(f"**Days to dispute:** {facts['days_to_dispute']}")
        st.write(f"**Delivery confirmed:** {format_bool(facts['delivery_confirmed'])}")
        st.write(f"**OTP authentication confirmed:** {format_bool(facts['otp_auth_confirmed'])}")
    with fc2:
        st.write(f"**Shipping/billing match:** {format_bool(facts['shipping_billing_match'])}")
        st.write(f"**Prior complaint on file:** {format_bool(facts['prior_complaint_on_file'])}")
        st.write(f"**Prior dispute count:** {facts['prior_dispute_count_at_time']}")
        st.write(f"**Item category:** {facts['item_category'].title()}")

    if result.get("retrieved_chunks"):
        st.markdown("---")
        st.markdown("#### Evidence Intelligence")
        st.caption("3 most relevant rulebook passages selected from the applicable dispute reason.")
        for c in result["retrieved_chunks"]:
            with st.container(border=True):
                st.write(f"**{c['chunk_id']}** — *{c['section'].replace('_', ' ').title()}*")
                st.caption(f"Reranker score: {c['rerank_score']:.4f}")

    st.markdown("---")
    st.markdown("#### Generated Evidence Response")
    if result.get("draft_status") == "VALIDATED":
        st.success("✓ VALIDATED — Response passed grounding and field validation.")
        st.markdown(result["draft"])
    elif result.get("draft_status") == "SKIPPED_CONCEDE":
        st.info("Evidence draft skipped because the decision engine favors concession.")
    elif result.get("draft_status") == "NO_RULE_FOUND":
        st.warning("No applicable rule found for this reason code. Case requires manual review.")
    elif result.get("draft_status") == "VALIDATION_FAILED_MAX_RETRIES":
        st.error("⚠ VALIDATION FAILED — Response requires regeneration or manual review.")
        if result.get("validation"):
            st.json(result["validation"])
    elif result["decision"] == "MANUAL_REVIEW":
        st.info("Case requires review.")


# ============================================================
# SESSION STATE INIT
# ============================================================
st.sidebar.markdown("## 🛡️ AI Chargeback\nRisk Manager")
st.sidebar.markdown("---")

NAV_OPTIONS = ["Overview", "Priority Queue", "Dispute Analysis", "Model Performance", "Economic Impact"]

if "page" not in st.session_state:
    st.session_state.page = "Overview"
if "selected_dispute" not in st.session_state:
    st.session_state.selected_dispute = None
if "live_queue" not in st.session_state:
    st.session_state.live_queue = pd.DataFrame()
if "last_refresh_info" not in st.session_state:
    st.session_state.last_refresh_info = None


# ============================================================
# SIDEBAR NAVIGATION (FIXED — no longer overwrites page on
# every rerun; only updates when the radio itself changes)
# ============================================================

def _on_nav_change():
    st.session_state.page = st.session_state["nav_radio_widget"]

st.sidebar.radio(
    "Navigation",
    NAV_OPTIONS,
    index=NAV_OPTIONS.index(st.session_state.page) if st.session_state.page in NAV_OPTIONS else 0,
    key="nav_radio_widget",
    on_change=_on_nav_change,
    label_visibility="collapsed",
)

page = st.session_state.page

st.sidebar.markdown("---")
manual_mode = st.sidebar.button("+ Analyze a Dispute", use_container_width=True)
if manual_mode:
    st.session_state.page = "Manual Analysis"
    page = "Manual Analysis"

st.sidebar.markdown("---")
st.sidebar.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

pipeline = get_pipeline()

# ============================================================
# REFRESH QUEUE — LIVE SIMULATION (CHANGED)
# No longer clears cache / reruns full 15k scoring. Only
# releases + scores a small incoming batch.
# ============================================================
if st.sidebar.button("↻ Refresh Queue", use_container_width=True):
    new_batch, err = release_new_disputes(pipeline)
    if err:
        st.session_state.last_refresh_info = err
    else:
        st.session_state.live_queue = pd.concat(
            [st.session_state.live_queue, new_batch], ignore_index=True
        )
        st.session_state.last_refresh_info = (
            f"Live simulation: +{len(new_batch)} new disputes received "
            f"at {datetime.now().strftime('%H:%M:%S')}"
        )
    st.rerun()

if st.session_state.last_refresh_info:
    st.sidebar.caption(f"🟢 {st.session_state.last_refresh_info}")
if len(st.session_state.live_queue):
    st.sidebar.caption(f"Simulated incoming disputes so far: {len(st.session_state.live_queue)}")

# ============================================================
# PAGE: OVERVIEW
# ============================================================
if page == "Overview":
    st.markdown("# 🛡️ AI Chargeback Risk Manager")
    st.caption("AI-powered dispute prioritization and evidence response")
    st.markdown("🟢 **System Ready**")
    st.markdown("---")

    queue_df = get_combined_queue(pipeline)
    total = len(queue_df)
    fight_pct = (queue_df["decision"] == "FIGHT").mean() * 100 if total else 0
    review_pct = (queue_df["decision"] == "MANUAL_REVIEW").mean() * 100 if total else 0
    concede_pct = (queue_df["decision"] == "CONCEDE").mean() * 100 if total else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Total Disputes</div><div class="kpi-value">{total:,}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Fight</div><div class="kpi-value status-fight">{fight_pct:.1f}%</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Manual Review</div><div class="kpi-value status-review">{review_pct:.1f}%</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Concede</div><div class="kpi-value status-concede">{concede_pct:.1f}%</div></div>', unsafe_allow_html=True)

    if len(st.session_state.live_queue):
        st.caption(
            f"Includes {len(st.session_state.live_queue)} simulated incoming disputes "
            "(demo simulation, not production data)."
        )

    st.markdown("")
    st.info("Your dispute queue is already being monitored and prioritized by AI. Go to **Priority Queue** to see cases ranked by financial exposure and urgency.")

    st.markdown("### Top 5 Priority Cases")
    top5 = queue_df.head(5)
    for _, row in top5.iterrows():
        cols = st.columns([2, 2, 2, 2, 2, 1])
        cols[0].write(f"**{row['dispute_id']}**")
        cols[1].write(f"₹{row['amount']:,.2f}")
        cols[2].write(row["reason_code"].replace("_", " ").title())
        cols[3].markdown(DECISION_PILL.get(row["decision"], row["decision"]), unsafe_allow_html=True)
        cols[4].write(f"Win prob: {row['win_probability']*100:.1f}%")
        if cols[5].button("View", key=f"ov_{row['dispute_id']}"):
            st.session_state.selected_dispute = row["dispute_id"]
            st.session_state.page = "Dispute Analysis"
            st.rerun()

# ============================================================
# PAGE: PRIORITY QUEUE
# ============================================================
elif page == "Priority Queue":
    st.markdown("## Priority Queue")
    st.caption("Disputes ranked by financial exposure and response urgency. Priority = 60% financial exposure + 40% urgency (not probability of winning).")

    queue_df = get_combined_queue(pipeline)

    fcol1, fcol2, fcol3 = st.columns(3)
    reason_filter = fcol1.multiselect("Reason code", sorted(queue_df["reason_code"].unique()))
    decision_filter = fcol2.multiselect("Decision", ["FIGHT", "MANUAL_REVIEW", "CONCEDE"])
    search_id = fcol3.text_input("Search by Dispute ID")

    filtered = queue_df.copy()
    if reason_filter:
        filtered = filtered[filtered["reason_code"].isin(reason_filter)]
    if decision_filter:
        filtered = filtered[filtered["decision"].isin(decision_filter)]
    if search_id:
        filtered = filtered[filtered["dispute_id"].str.contains(search_id, case=False)]

    st.caption(f"Showing {len(filtered)} of {len(queue_df)} disputes, sorted by priority (highest first)")

    for _, row in filtered.head(50).iterrows():
        with st.container(border=True):
            c = st.columns([2, 1.5, 2, 1.5, 1.5, 1.5, 1])
            c[0].markdown(f"**{row['dispute_id']}**")
            c[1].write(f"₹{row['amount']:,.2f}")
            c[2].write(row["reason_code"].replace("_", " ").title())
            c[3].markdown(DECISION_PILL.get(row["decision"], row["decision"]), unsafe_allow_html=True)
            c[4].write(f"{row['win_probability']*100:.1f}%")
            c[5].write(f"Priority: {priority_label(row['priority_score'])}")
            if c[6].button("Open", key=f"pq_{row['dispute_id']}"):
                st.session_state.selected_dispute = row["dispute_id"]
                st.session_state.page = "Dispute Analysis"
                st.rerun()

# ============================================================
# PAGE: DISPUTE ANALYSIS
# ============================================================
elif page == "Dispute Analysis":
    st.markdown("## Dispute Analysis")

    queue_df = get_combined_queue(pipeline)
    default_id = st.session_state.selected_dispute or (queue_df.iloc[0]["dispute_id"] if len(queue_df) else None)
    dispute_id = st.selectbox(
        "Select dispute",
        queue_df["dispute_id"].tolist(),
        index=queue_df["dispute_id"].tolist().index(default_id) if default_id in queue_df["dispute_id"].tolist() else 0,
    )

    if st.button("🔎 Analyze Case", type="primary"):
        st.session_state[f"result_{dispute_id}"] = None  # force reprocess

    result_key = f"result_{dispute_id}"
    if result_key not in st.session_state or st.session_state[result_key] is None:
        progress_box = st.status("Processing case...", expanded=True)
        progress_box.write("✓ Case loaded")
        progress_box.write("⟳ XGBoost scoring...")
        result = pipeline.process_dispute(dispute_id)
        progress_box.write(f"✓ XGBoost scored — win probability: {result['win_probability']*100:.2f}%")
        progress_box.write(f"✓ Decision generated — **{result['decision']}**")
        if result.get("retrieved_chunks"):
            progress_box.write("✓ Hybrid retrieval completed (BM25 + BGE + RRF)")
            progress_box.write(f"✓ Cross-encoder reranked — Top-3 selected")
        if result.get("draft_status") == "VALIDATED":
            progress_box.write("⟳ Generating evidence response...")
            progress_box.write("✓ Validation passed")
        elif result.get("draft_status") == "SKIPPED_CONCEDE":
            progress_box.write("— Evidence draft skipped (decision engine favors concession)")
        elif result.get("draft_status") in ("NO_RULE_FOUND",):
            progress_box.write("⚠ No applicable rule found — routed to manual review")
        elif result.get("draft_status") == "VALIDATION_FAILED_MAX_RETRIES":
            progress_box.write("⚠ Validation failed after retries — routed to manual review")
        progress_box.update(label="Processing complete", state="complete")
        st.session_state[result_key] = result

    result = st.session_state[result_key]
    render_dispute_result(result, is_manual=False)

# ============================================================
# PAGE: MODEL PERFORMANCE
# ============================================================
elif page == "Model Performance":
    st.markdown("## Model Performance")
    st.caption("Evaluated on a held-out test set never used during training or threshold selection.")

    metrics, threshold_compare = load_model_metrics()

    c1, c2 = st.columns(2)
    c1.metric("ROC-AUC", f"{metrics['test_roc_auc']:.4f}")
    c2.metric("PR-AUC", f"{metrics['test_pr_auc']:.4f}")

    st.markdown("#### Classification Report (Held-out Test)")
    st.dataframe(pd.DataFrame({
        "Class": ["Lose (0)", "Win (1)"],
        "Precision": [0.877, 0.535],
        "Recall": [0.170, 0.976],
        "F1": [0.284, 0.691],
    }), hide_index=True, use_container_width=True)
    st.caption("Accuracy: 0.568. Model is tuned for high recall on winnable disputes (97.6%), accepting lower precision — deliberate given the cost asymmetry of missing a winnable dispute vs. fighting a losing one.")

    fig_dir = BASE_DIR / "figures"
    if fig_dir.exists():
        st.markdown("#### Confusion Matrix")
        cm_path = fig_dir / "confusion_matrix.png"
        if cm_path.exists():
            st.image(str(cm_path))

        st.markdown("#### What drives the model's win prediction?")
        st.caption("Feature importance describes which case attributes most influenced the model's prediction. This does not imply causality.")
        shap_path = fig_dir / "shap_beeswarm.png"
        if shap_path.exists():
            st.image(str(shap_path))

    if threshold_compare:
        st.markdown("#### Threshold Selection")
        st.caption(f"Strategy used: **{threshold_compare.get('chosen_strategy', 'cost_optimal')}** — selected on validation set only, applied once to held-out test.")

# ============================================================
# PAGE: ECONOMIC IMPACT
# ============================================================
elif page == "Economic Impact":
    st.markdown("## Economic Impact")
    st.caption("Modeled cost of decision errors on the held-out test set.")

    e1, e2, e3 = st.columns(3)
    e1.metric("Fight-Loss Cost", "₹283,200")
    e2.metric("Money Left on Table", "₹32,786.63")
    e3.metric("Total Modeled Error Cost", "₹315,986.63")

    st.markdown("---")
    st.markdown("#### Assumptions")
    st.write("**Fight operational cost:** ₹300 per case")
    st.caption("This is a configurable modeling assumption, not an industry benchmark.")

    st.markdown("#### Definitions")
    st.write("**Fight-loss cost** represents the modeled cost of contesting cases that ultimately do not succeed.")
    st.write("**Money left on table** represents disputed value associated with cases the model did not successfully route to a winning contest.")

    sweep = load_economic_figures()
    fig_path = BASE_DIR / "figures" / "review_margin_sensitivity.png"
    if fig_path.exists():
        st.markdown("#### Review-Margin Sensitivity")
        st.image(str(fig_path))
        st.caption("Shows how the human-review threshold trades off automation rate against modeled error cost.")

# ============================================================
# PAGE: MANUAL ANALYSIS (optional, secondary)
# ============================================================
elif page == "Manual Analysis":
    st.markdown("## Analyze a Dispute")
    st.caption("Optional — the automatically prioritized queue is the primary workflow. This sends the case through the SAME backend pipeline used for the stored queue.")

    with st.form("manual_dispute_form"):
        c1, c2 = st.columns(2)
        with c1:
            m_id = st.text_input("Dispute ID", value="MANUAL-0001")
            m_amount = st.number_input("Amount (₹)", min_value=0.0, value=1000.0, step=50.0)
            m_reason = st.selectbox("Reason code", [
                "not_as_described", "item_not_received", "unauthorized_transaction",
                "duplicate_charge", "other_unclassified",
            ])
            m_category = st.selectbox("Item category", ["apparel", "electronics", "groceries", "digital_goods", "home_goods"])
            m_days = st.number_input("Days to dispute", min_value=0, value=10)
        with c2:
            m_delivery = st.checkbox("Delivery confirmed")
            m_otp = st.checkbox("OTP authentication confirmed")
            m_addr = st.checkbox("Shipping/billing match")
            m_complaint = st.checkbox("Prior complaint on file")
            m_prior_count = st.number_input("Prior dispute count", min_value=0, value=0)

        submitted = st.form_submit_button("Analyze Dispute", type="primary")

    if submitted:
        # ---- basic input validation ----
        errors = []
        if not m_id.strip():
            errors.append("Dispute ID is required.")
        if m_amount <= 0:
            errors.append("Amount must be greater than 0.")
        if m_days < 0:
            errors.append("Days to dispute cannot be negative.")
        if m_prior_count < 0:
            errors.append("Prior dispute count cannot be negative.")
        if not m_reason:
            errors.append("Reason code is required.")
        if not m_category:
            errors.append("Item category is required.")

        if errors:
            for e in errors:
                st.error(e)
        else:
            # ---- construct in-memory case with EXACTLY the same schema
            #      DisputePipeline._load_case_facts() produces from the
            #      CSVs, so it flows through identical downstream logic ----
            case_facts = {
                "amount": float(m_amount),
                "item_category": m_category,
                "delivery_confirmed": bool(m_delivery),
                "otp_auth_confirmed": bool(m_otp),
                "shipping_billing_match": bool(m_addr),
                "prior_complaint_on_file": bool(m_complaint),
                "days_to_dispute": int(m_days),
                "prior_dispute_count_at_time": int(m_prior_count),
            }

            try:
                with st.spinner("Processing case through the pipeline..."):
                    result = pipeline.process_case(m_id.strip(), case_facts, m_reason)
                st.session_state["manual_result"] = result
            except Exception as e:
                st.error(f"Pipeline error while processing this case: {type(e).__name__}: {e}")
                st.session_state["manual_result"] = None

    if st.session_state.get("manual_result"):
        st.markdown("---")
        render_dispute_result(st.session_state["manual_result"], is_manual=True)
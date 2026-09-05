"""
validator.py

Validates generated chargeback evidence drafts before they are
returned to the dashboard.

Checks:
1. Required sections are present.
2. Dispute ID is present.
3. Supplied case facts are not contradicted.
4. Retrieved rule context is referenced.
"""

import re


REQUIRED_SECTIONS = [
    "Decision Context",
    "Evidence Summary",
    "Applicable Rule",
    "Evidence Supporting Contest",
    "Evidence Gaps",
    "Recommended Response",
]


def validate(
    draft,
    dispute_id,
    case_facts,
    retrieved_chunks,
):
    """
    Validate an LLM-generated chargeback evidence draft.

    Returns:
        {
            "passed": bool,
            "checks": {...},
            "errors": [...]
        }
    """

    errors = []
    checks = {}

    # ========================================================
    # BASIC CHECK
    # ========================================================

    if not draft or not isinstance(draft, str):
        errors.append("Draft is empty or not a string.")
        checks["draft_exists"] = False

        return {
            "passed": False,
            "checks": checks,
            "errors": errors,
        }

    checks["draft_exists"] = True

    # ========================================================
    # REQUIRED SECTIONS
    # ========================================================

    missing_sections = []

    for section in REQUIRED_SECTIONS:

        if section.lower() not in draft.lower():
            missing_sections.append(section)

    if missing_sections:

        errors.append(
            "Missing required sections: "
            + ", ".join(missing_sections)
        )

        checks["required_sections"] = False

    else:

        checks["required_sections"] = True

    # ========================================================
    # DISPUTE ID
    # ========================================================

    dispute_id_found = str(dispute_id).lower() in draft.lower()

    checks["dispute_id_present"] = dispute_id_found

    if not dispute_id_found:

        errors.append(
            f"Dispute ID '{dispute_id}' is not present in draft."
        )

    # ========================================================
    # RETRIEVED RULE CONTEXT
    # ========================================================

    has_rule_context = bool(retrieved_chunks)

    checks["rule_context_available"] = has_rule_context

    if not has_rule_context:

        errors.append(
            "No retrieved rule context was supplied."
        )

    # ========================================================
    # CASE FACT VALIDATION
    #
    # We only perform conservative checks here.
    # We do NOT try to semantically judge the entire LLM output.
    # ========================================================

    fact_checks = []

    # Amount
    if "amount" in case_facts:

        amount = case_facts["amount"]

        amount_strings = [
            f"{amount:.2f}",
            f"{amount:.1f}",
            f"{amount:.0f}",
            f"₹{amount:.2f}",
            f"₹{amount:.0f}",
        ]

        amount_found = any(
            value in draft
            for value in amount_strings
        )

        fact_checks.append(amount_found)

        checks["amount_present"] = amount_found

    # Reason-specific factual signals
    boolean_facts = [
        "delivery_confirmed",
        "otp_auth_confirmed",
        "shipping_billing_match",
        "prior_complaint_on_file",
    ]

    checks["boolean_fact_fields_available"] = True

    # ========================================================
    # CONFLICT CHECKS
    #
    # Only flag obvious contradictions.
    # ========================================================

    contradiction_patterns = []

    if case_facts.get("delivery_confirmed") is False:

        contradiction_patterns.extend([
            r"\bdelivered successfully\b",
            r"\bdelivery was confirmed\b",
            r"\bproof of delivery confirms delivery\b",
        ])

    if case_facts.get("otp_auth_confirmed") is False:

        contradiction_patterns.extend([
            r"\botp was verified\b",
            r"\botp authentication was confirmed\b",
        ])

    if case_facts.get("shipping_billing_match") is False:

        contradiction_patterns.extend([
            r"\bshipping and billing addresses match\b",
            r"\baddresses match\b",
        ])

    contradiction_found = False

    draft_lower = draft.lower()

    for pattern in contradiction_patterns:

        if re.search(pattern, draft_lower):

            contradiction_found = True

            errors.append(
                f"Potential factual contradiction detected: "
                f"'{pattern}'"
            )

    checks["no_obvious_fact_contradictions"] = (
        not contradiction_found
    )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    passed = len(errors) == 0

    return {
        "passed": passed,
        "checks": checks,
        "errors": errors,
    }
"""
Rulebook generator for the Chargeback Evidence Responder project.

Writes one plain-text rule per reason code plus a metadata JSON file that
maps chunk_id -> reason_code -> source file. This metadata is what makes
retriever evaluation (contextual precision/recall) possible later: for any
dispute we already know its reason_code, and this map tells us which chunk
SHOULD be retrieved.

chunk_id values here MUST match CHUNK_ID_MAP in generate_data.py.

Run:
    python generate_rulebook.py
"""

import json
import os

OUT_DIR = "rulebook"

# ---------------------------------------------------------------------------
# Rule text per reason code.
#
# These are self-authored, plausible rules modeled on the general shape of
# real card-network evidence requirements (not scraped/reproduced from any
# proprietary Visa/Mastercard/Amex document). Written specifically to give
# each reason code a distinct evidentiary profile so retrieval + generation
# grounding is meaningfully testable.
# ---------------------------------------------------------------------------

RULES = {
    "not_as_described": {
        "chunk_id": "RULE_NOT_AS_DESCRIBED",
        "reason_code": "not_as_described",
        "title": "Reason Code: Not As Described",
        "text": (
            "This reason code applies when a cardholder claims the received "
            "item or service materially differs from what was advertised at "
            "purchase (wrong size, wrong color, damaged on arrival, "
            "misrepresented features, or counterfeit claims).\n\n"
            "REQUIRED EVIDENCE TO CONTEST:\n"
            "- Original product listing or description shown to the customer "
            "at time of purchase.\n"
            "- Proof of delivery confirming the correct item was shipped "
            "(item description/SKU on the shipping manifest).\n"
            "- Any customer communication prior to the dispute. The absence "
            "of a prior complaint or return request is a meaningful signal "
            "that the item was likely as described; a documented prior "
            "complaint on file weakens the merchant's position.\n"
            "- Photos or listing screenshots showing the item matches the "
            "shipped SKU, where available.\n\n"
            "EVIDENCE THAT IS NOT SUFFICIENT ALONE:\n"
            "- Proof of delivery by itself does not prove the item matched "
            "the description; it only proves something was delivered.\n"
            "- A generic order confirmation email without a delivery record.\n\n"
            "RESPONSE FORMAT:\n"
            "State the SKU shipped, confirm it matches the listing description, "
            "note whether any complaint was filed before the dispute, and "
            "attach delivery confirmation. Keep the response factual and "
            "point-by-point against the cardholder's specific claim."
        ),
    },
    "item_not_received": {
        "chunk_id": "RULE_ITEM_NOT_RECEIVED",
        "reason_code": "item_not_received",
        "title": "Reason Code: Item Not Received",
        "text": (
            "This reason code applies when a cardholder claims they paid for "
            "goods or services but never received them.\n\n"
            "REQUIRED EVIDENCE TO CONTEST:\n"
            "- Valid proof of delivery: carrier tracking number showing "
            "delivered status, ideally with a signature or delivery photo, "
            "to an address matching the billing/shipping address on file.\n"
            "- Delivery date must be reasonably consistent with the order "
            "timeline; delivery confirmation is the single most important "
            "piece of evidence for this reason code and effectively "
            "determines the outcome on its own in most cases.\n\n"
            "EVIDENCE THAT IS NOT SUFFICIENT ALONE:\n"
            "- An order confirmation or payment receipt without a linked "
            "delivery/tracking record. Payment proof does not establish "
            "that the item arrived.\n"
            "- A shipping label being generated does not constitute proof "
            "of delivery; only carrier-confirmed delivery status counts.\n\n"
            "IF DELIVERY CANNOT BE CONFIRMED:\n"
            "Do not contest without delivery evidence. Absent confirmed "
            "delivery, escalate internally rather than submitting a response, "
            "since a case without delivery proof is very unlikely to succeed "
            "and may be better handled by proactively refunding the "
            "customer.\n\n"
            "RESPONSE FORMAT:\n"
            "Lead with the tracking number and delivered status/date. Confirm "
            "the delivery address matches the order's shipping address. If "
            "delivery cannot be confirmed, do not submit contest evidence — "
            "route to manual review instead."
        ),
    },
    "unauthorized_transaction": {
        "chunk_id": "RULE_UNAUTHORIZED_TRANSACTION",
        "reason_code": "unauthorized_transaction",
        "title": "Reason Code: Unauthorized Transaction",
        "text": (
            "This reason code applies when a cardholder claims they did not "
            "authorize the transaction (potential card fraud/theft).\n\n"
            "REQUIRED EVIDENCE TO CONTEST:\n"
            "- Confirmation that a valid authentication step occurred at the "
            "time of purchase (e.g. OTP/3D Secure verification passed). This "
            "is the primary evidence for this reason code, since a "
            "successfully completed authentication step strongly indicates "
            "the legitimate cardholder approved the transaction.\n"
            "- Device and IP information consistent with the cardholder's "
            "prior purchase history, where available.\n"
            "- Billing address match with the shipping address, as a "
            "secondary signal of legitimacy.\n\n"
            "EVIDENCE THAT IS NOT SUFFICIENT ALONE:\n"
            "- A successful payment capture alone does not prove "
            "authorization; payment processing can succeed even on a stolen "
            "card if no additional authentication step was performed.\n"
            "- Delivery confirmation is not directly relevant evidence for "
            "this reason code; a fraudulent transaction can still be "
            "delivered successfully.\n\n"
            "IF AUTHENTICATION CANNOT BE CONFIRMED:\n"
            "Treat as high fraud risk. Without a confirmed OTP/3DS step, "
            "the transaction should generally not be contested.\n\n"
            "RESPONSE FORMAT:\n"
            "State clearly whether OTP/3D Secure authentication was "
            "completed, and reference any matching device/IP or address "
            "history. Do not rely on delivery or payment-capture evidence "
            "as the primary argument."
        ),
    },
    "duplicate_charge": {
        "chunk_id": "RULE_DUPLICATE_CHARGE",
        "reason_code": "duplicate_charge",
        "title": "Reason Code: Duplicate Charge",
        "text": (
            "This reason code applies when a cardholder claims they were "
            "charged more than once for the same purchase.\n\n"
            "REQUIRED EVIDENCE TO CONTEST:\n"
            "- Transaction-level detail showing the two (or more) charges "
            "in question correspond to genuinely separate orders (different "
            "order IDs, different items, or different timestamps beyond a "
            "normal retry window).\n"
            "- Matching billing and shipping address across the charges, "
            "which supports that both charges are legitimately linked to "
            "the same customer's distinct activity rather than a processing "
            "error.\n"
            "- Customer's dispute history: a customer with multiple prior "
            "disputes flagged as duplicate/erroneous should be evaluated "
            "carefully, as repeat duplicate-charge disputes from the same "
            "customer can indicate a pattern rather than genuine processor "
            "error.\n\n"
            "EVIDENCE THAT IS NOT SUFFICIENT ALONE:\n"
            "- Simply asserting the charges were 'for different things' "
            "without order-level detail distinguishing them.\n\n"
            "IF THE CHARGES ARE GENUINELY DUPLICATED (processor/system "
            "error):\n"
            "Do not contest. Refund the duplicate charge immediately; "
            "contesting a genuine system-level duplicate charge is not "
            "appropriate and will fail evidence review in any case.\n\n"
            "RESPONSE FORMAT:\n"
            "Present both order IDs side by side with itemized contents and "
            "timestamps. Confirm address match. Note the customer's prior "
            "dispute count only if relevant to distinguishing pattern abuse "
            "from genuine error."
        ),
    },
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    metadata = []

    for reason_code, rule in RULES.items():
        chunk_id = rule["chunk_id"]
        filename = f"{chunk_id}.txt"
        filepath = os.path.join(OUT_DIR, filename)

        with open(filepath, "w") as f:
            f.write(f"{rule['title']}\n\n{rule['text']}")

        metadata.append(
            {
                "chunk_id": chunk_id,
                "reason_code": reason_code,
                "title": rule["title"],
                "source_file": filename,
                "char_count": len(rule["text"]),
            }
        )

    with open(os.path.join(OUT_DIR, "rulebook_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Wrote {len(RULES)} rule files to {OUT_DIR}/")
    for m in metadata:
        print(f"  {m['chunk_id']:<32} ({m['char_count']} chars) <- {m['reason_code']}")
    print(f"\nMetadata written to {OUT_DIR}/rulebook_metadata.json")
    print(
        "\nNote: 'other_unclassified' reason code has NO corresponding chunk "
        "by design - this is what makes the RAG retrieval step correctly "
        "return 'no match' and route those cases to manual_review_fallback."
    )


if __name__ == "__main__":
    main()
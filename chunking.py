"""
chunking.py — Semantic/section-based chunker for the reason-code rulebook.

Each RULE_*.txt file is split by its known section headers (the rules were
authored with consistent structure), not by fixed character windows. Within 
a section, if it exceeds ~120 words, it's further split with a 20-word
overlap so no chunk exceeds the target size while still respecting section
boundaries (a chunk never spans two sections).

Output: rulebook/chunks.jsonl — one JSON object per chunk:
  {
    "chunk_id": "not_as_described_required_evidence_0",
    "reason_code": "not_as_described",
    "section": "required_evidence",
    "source": "RULE_NOT_AS_DESCRIBED.txt",
    "text": "..."
  }

Run:
    python chunking.py
"""

import json
import os
import re

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def deco(fn):
            return fn
        return deco if args and callable(args[0]) is False else (args[0] if args else deco)


RULEBOOK_DIR = "rulebook"
CHUNK_SIZE_WORDS = 120
CHUNK_OVERLAP_WORDS = 20


# Maps the ALL-CAPS section markers used in generate_rulebook.py's rule text
# to a normalized section key. Order matters for splitting.
SECTION_MARKERS = [
    ("REQUIRED EVIDENCE TO CONTEST:", "required_evidence"),
    ("EVIDENCE THAT IS NOT SUFFICIENT ALONE:", "insufficient_evidence"),
    ("IF DELIVERY CANNOT BE CONFIRMED:", "exception_do_not_contest"),
    ("IF AUTHENTICATION CANNOT BE CONFIRMED:", "exception_do_not_contest"),
    ("IF THE CHARGES ARE GENUINELY DUPLICATED", "exception_do_not_contest"),
    ("RESPONSE FORMAT:", "response_format"),
]


def split_words(text, size, overlap):
    """Split text into overlapping word-count windows.

    Sections remain independent; chunks may split at word boundaries.
    """
    words = text.split()

    if len(words) <= size:
        return [text.strip()]

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + size, len(words))

        chunks.append(
            " ".join(words[start:end]).strip()
        )

        if end == len(words):
            break

        start = end - overlap

    return chunks


@traceable(name="chunk_rulebook_file")
def chunk_file(filepath, reason_code):
    with open(filepath, encoding="cp1252") as f:
        raw = f.read()

    # Drop the title line (first line + blank line)
    body = raw.split("\n\n", 1)[1] if "\n\n" in raw else raw

    # Find section boundaries
    positions = []

    for marker, key in SECTION_MARKERS:
        idx = body.find(marker)

        if idx != -1:
            positions.append((idx, marker, key))

    positions.sort()

    # Applicability = everything before the first marker
    sections = []

    if positions:
        applicability_text = body[:positions[0][0]].strip()

        sections.append(
            ("applicability", applicability_text)
        )

        for i, (idx, marker, key) in enumerate(positions):
            end = (
                positions[i + 1][0]
                if i + 1 < len(positions)
                else len(body)
            )

            section_text = body[
                idx + len(marker):end
            ].strip()

            sections.append(
                (key, section_text)
            )

    else:
        sections.append(
            ("full_text", body.strip())
        )

    chunks = []

    for section_key, section_text in sections:

        if not section_text:
            continue

        sub_chunks = split_words(
            section_text,
            CHUNK_SIZE_WORDS,
            CHUNK_OVERLAP_WORDS
        )

        for i, sub in enumerate(sub_chunks):

            suffix = (
                f"_{i}"
                if len(sub_chunks) > 1
                else ""
            )

            chunk_id = (
                f"{reason_code}_{section_key}{suffix}"
            )

            # Keep the semantic section information inside the text
            # so the embedding/reranker sees it as well as metadata.
            chunk_text = (
                f"Reason Code: {reason_code}\n"
                f"Section: {section_key}\n\n"
                f"{sub}"
            )

            chunks.append({
                "chunk_id": chunk_id,
                "reason_code": reason_code,
                "section": section_key,
                "source": os.path.basename(filepath),
                "text": chunk_text.strip(),
                "word_count": len(chunk_text.split()),
            })

    return chunks


def main():
    with open(
        f"{RULEBOOK_DIR}/rulebook_metadata.json",
        encoding="utf-8"
    ) as f:
        metadata = json.load(f)

    all_chunks = []

    for entry in metadata:
        filepath = (
            f"{RULEBOOK_DIR}/{entry['source_file']}"
        )

        chunks = chunk_file(
            filepath,
            entry["reason_code"]
        )

        all_chunks.extend(chunks)

        print(
            f"{entry['reason_code']:<28} -> "
            f"{len(chunks)} chunks"
        )

    out_path = f"{RULEBOOK_DIR}/chunks.jsonl"

    with open(
        out_path,
        "w",
        encoding="utf-8"
    ) as f:
        for chunk in all_chunks:
            f.write(
                json.dumps(
                    chunk,
                    ensure_ascii=False
                ) + "\n"
            )

    print(f"\nTotal chunks: {len(all_chunks)}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
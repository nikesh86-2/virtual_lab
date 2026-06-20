"""
validate_dataset.py

Validates and cleans your JSONL training data before fine-tuning.

Filters:
- empty / corrupt entries
- low-information outputs
- malformed instruction structure
- duplicates
"""

import json

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
MIN_OUTPUT_LENGTH = 30
MAX_REPEAT_THRESHOLD = 0.3  # repeated characters ratio


# ----------------------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------------------
def load_jsonl(path):
    data = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except Exception:
                continue

    return data


# ----------------------------------------------------------------------
# BASIC VALIDATION
# ----------------------------------------------------------------------
def is_valid_entry(entry):
    """Basic structural validation"""

    if not isinstance(entry, dict):
        return False

    if "instruction" not in entry:
        return False

    if "output" not in entry:
        return False

    if not entry["output"]:
        return False

    if len(entry["output"]) < MIN_OUTPUT_LENGTH:
        return False

    return True


# ----------------------------------------------------------------------
# QUALITY FILTER
# ----------------------------------------------------------------------
def is_low_quality(text):
    """Detect low-information outputs"""

    text = text.strip()

    if not text:
        return True

    # repeated character spam
    repeats = max(text.count(c) for c in set(text)) / max(len(text), 1)
    if repeats > MAX_REPEAT_THRESHOLD:
        return True

    # generic failure patterns
    bad_patterns = [
        "failed",
        "error",
        "unknown",
        "could not",
        "n/a",
    ]

    lower = text.lower()

    if any(p in lower for p in bad_patterns):
        return True

    return False


# ----------------------------------------------------------------------
# DEDUPLICATION
# ----------------------------------------------------------------------
def deduplicate(data):
    seen = set()
    filtered = []

    for item in data:
        key = (
            item.get("instruction", "") +
            item.get("input", "") +
            item.get("output", "")[:100]
        )

        if key not in seen:
            seen.add(key)
            filtered.append(item)

    return filtered


# ----------------------------------------------------------------------
# SCORING (OPTIONAL)
# ----------------------------------------------------------------------
def score_entry(entry):
    score = 0

    output = entry.get("output", "")

    # reward detail
    if len(output) > 200:
        score += 1

    # reward structure
    if ":" in output or "\n" in output:
        score += 1

    # reward scientific content
    keywords = ["structure", "binding", "energy", "rna", "protein"]

    if any(k in output.lower() for k in keywords):
        score += 1

    return score


# ----------------------------------------------------------------------
# MAIN CLEANING PIPELINE
# ----------------------------------------------------------------------
def clean_dataset(input_path, output_path, min_score=1):
    print(f"\nLoading: {input_path}")

    data = load_jsonl(input_path)
    print(f"Loaded {len(data)} entries")

    # Step 1: structure validation
    valid = [d for d in data if is_valid_entry(d)]
    print(f"After validation: {len(valid)}")

    # Step 2: remove low quality
    filtered = [d for d in valid if not is_low_quality(d["output"])]
    print(f"After quality filter: {len(filtered)}")

    # Step 3: deduplicate
    deduped = deduplicate(filtered)
    print(f"After deduplication: {len(deduped)}")


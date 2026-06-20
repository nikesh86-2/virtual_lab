from __future__ import annotations

import logging
import os

import yaml


log = logging.getLogger("virtual_lab")


def load_topics(path: str = "research_topics.yaml") -> list:
    """
    Load topic definitions from YAML.
    """
    if not os.path.exists(path):
        log.warning("Topic file %s not found. Using default topic.", path)
        return []

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    return data.get("topics", [])


def select_topic(topics: list, index: int | None = None) -> dict:
    """
    Select topic by index, interactive prompt, or fallback.
    """
    if not topics:
        return {
            "name": "viral packaging motifs",
            "description": "Identify cis-acting RNA packaging signals in viral systems",
            "seed_questions": [],
        }

    if index is not None:
        if 0 <= index < len(topics):
            return topics[index]
        return topics[0]

    print("\n=== Available Research Topics ===")

    for i, topic in enumerate(topics):
        print(f"[{i}] {topic['name']}: {topic['description']}")

    print("[-1] Custom topic")

    try:
        choice = int(input("\nSelect topic number: "))
    except Exception:
        choice = 0

    if choice == -1:
        return {
            "name": input("Topic name: "),
            "description": input("Description: "),
            "seed_questions": [input("Initial question: ")],
        }

    return topics[max(0, min(choice, len(topics) - 1))]
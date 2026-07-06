from __future__ import annotations

import logging
import os
from typing import Any

import yaml


log = logging.getLogger("virtual_lab")


def _normalise_topic(topic: dict[str, Any]) -> dict[str, Any]:
    """
    Normalise topic metadata so downstream agents always receive consistent fields.

    Important for topic-reactive literature streaming:
      - researcher_agent uses virus_family / virus_genus / virus_name
      - LiteratureMemory stores topic_name / research_topic / task_type
      - postrun export uses these fields for training provenance
    """
    if not isinstance(topic, dict):
        topic = {}

    name = str(topic.get("name") or topic.get("topic_name") or "").strip()
    description = str(
        topic.get("description")
        or topic.get("topic_description")
        or topic.get("research_topic")
        or ""
    ).strip()

    seed_questions = topic.get("seed_questions") or []
    if isinstance(seed_questions, str):
        seed_questions = [seed_questions]
    elif not isinstance(seed_questions, list):
        seed_questions = []

    virus_family = str(topic.get("virus_family") or "").strip()
    virus_genus = str(topic.get("virus_genus") or "").strip()
    virus_name = str(topic.get("virus_name") or "").strip()

    normalised = dict(topic)
    normalised["name"] = name or "viral packaging motifs"
    normalised["topic_name"] = normalised["name"]
    normalised["description"] = (
        description
        or "Identify cis-acting RNA packaging signals in viral systems"
    )
    normalised["topic_description"] = normalised["description"]
    normalised["research_topic"] = (
        topic.get("research_topic")
        or normalised["name"]
    )
    normalised["seed_questions"] = seed_questions
    normalised["virus_family"] = virus_family
    normalised["virus_genus"] = virus_genus
    normalised["virus_name"] = virus_name

    return normalised


def load_topics(path: str = "research_topics.yaml") -> list[dict[str, Any]]:
    """
    Load topic definitions from YAML and normalise topic metadata.
    """
    if not os.path.exists(path):
        log.warning("Topic file %s not found. Using default topic.", path)
        return []

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    topics = data.get("topics", []) or []

    if not isinstance(topics, list):
        log.warning("Topic file %s has non-list 'topics' field; using default.", path)
        return []

    return [_normalise_topic(t) for t in topics if isinstance(t, dict)]


def select_topic(topics: list, index: int | None = None) -> dict:
    """
    Select topic by index, interactive prompt, or fallback.

    Also respects VLAB_TOPIC_INDEX if index is not passed.
    """
    if not topics:
        return _normalise_topic(
            {
                "name": "viral packaging motifs",
                "description": "Identify cis-acting RNA packaging signals in viral systems",
                "seed_questions": [],
                "virus_family": "",
                "virus_genus": "",
                "virus_name": "",
            }
        )

    if index is None:
        env_index = os.getenv("VLAB_TOPIC_INDEX", "").strip()
        if env_index:
            try:
                index = int(env_index)
            except ValueError:
                index = None

    if index is not None:
        if 0 <= index < len(topics):
            return _normalise_topic(topics[index])
        return _normalise_topic(topics[0])

    print("\n=== Available Research Topics ===")

    for i, topic in enumerate(topics):
        print(f"[{i}] {topic.get('name', 'Unnamed')}: {topic.get('description', '')}")

    print("[-1] Custom topic")

    try:
        choice = int(input("\nSelect topic number: "))
    except Exception:
        choice = 0

    if choice == -1:
        custom = {
            "name": input("Topic name: "),
            "description": input("Description: "),
            "virus_family": input("Virus family, optional: "),
            "virus_genus": input("Virus genus, optional: "),
            "virus_name": input("Virus name, optional: "),
            "seed_questions": [input("Initial question: ")],
        }
        return _normalise_topic(custom)

    return _normalise_topic(topics[max(0, min(choice, len(topics) - 1))])
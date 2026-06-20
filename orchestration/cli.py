from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from VLAB2.core.training_data_collector import TrainingDataCollector

from VLAB2.orchestration.config import bootstrap_runtime
from VLAB2.orchestration.orchestrator import virtual_lab
from VLAB2.orchestration.postrun import bootstrap_knowledge_base, run_postrun_pipeline
from VLAB2.orchestration.reporting import print_user_friendly_summary
from VLAB2.orchestration.state_factory import build_initial_state
from VLAB2.orchestration.state_schema import safe_jsonable
from VLAB2.orchestration.topics import load_topics, select_topic
from VLAB2.orchestration.wrappers import build_wrapper_bundle


log = logging.getLogger("virtual_lab")


RUNTIME_ONLY_KEYS = {
    "wrappers",
    "data_collector",
    "_md_cache",
}


def _topic_output_name(topic_name: str) -> str:
    """
    Build stable output filename from topic name.
    """
    safe = topic_name.replace(" ", "_").replace("/", "_").lower()
    return f"lab_results_{safe}.json"


def _fallback_jsonable(obj: Any) -> Any:
    """
    Conservative JSON converter.

    This is used only if state_schema.safe_jsonable returns None.
    It prevents runtime Python objects from collapsing the entire final state
    into JSON null.
    """
    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, Mapping):
        out = {}

        for k, v in obj.items():
            key = str(k)

            if key in RUNTIME_ONLY_KEYS:
                continue

            out[key] = _fallback_jsonable(v)

        return out

    if isinstance(obj, (list, tuple, set)):
        return [_fallback_jsonable(v) for v in obj]

    # LangChain messages and other objects sometimes expose dict/model_dump.
    if hasattr(obj, "model_dump"):
        try:
            return _fallback_jsonable(obj.model_dump())
        except Exception:
            pass

    if hasattr(obj, "dict"):
        try:
            return _fallback_jsonable(obj.dict())
        except Exception:
            pass

    # Last resort: keep a useful string instead of failing the whole save.
    return repr(obj)


def _strip_runtime_objects(final_result: Any) -> Any:
    """
    Remove runtime-only objects from final state before JSON serialisation.
    """
    if not isinstance(final_result, Mapping):
        return final_result

    cleaned = {}

    for k, v in dict(final_result).items():
        key = str(k)

        if key in RUNTIME_ONLY_KEYS:
            log.info("Skipping runtime-only final-state key during JSON save: %s", key)
            continue

        cleaned[key] = v

    return cleaned


def _coerce_final_result_for_json(final_result: Any) -> Any:
    """
    Convert final graph result into a JSON-safe payload.

    Important:
      - Refuses to silently save None/null.
      - Strips runtime objects such as wrappers/data_collector.
      - Falls back to a local conservative serializer if safe_jsonable returns None.
    """
    if final_result is None:
        raise RuntimeError(
            "Final graph result is None. Refusing to write JSON null. "
            "This usually means the graph/orchestrator did not return a final state."
        )

    raw_payload = _strip_runtime_objects(final_result)

    if isinstance(raw_payload, Mapping):
        raw_payload = dict(raw_payload)

    payload = safe_jsonable(raw_payload)

    if payload is None:
        log.warning(
            "safe_jsonable(...) returned None for final_result type %s; "
            "falling back to local JSON sanitiser.",
            type(final_result).__name__,
        )
        payload = _fallback_jsonable(raw_payload)

    if payload is None:
        raise RuntimeError(
            "Final payload is still None after fallback sanitisation. "
            "Refusing to write JSON null."
        )

    if isinstance(payload, dict) and not payload:
        raise RuntimeError(
            "Final result payload is an empty dict. Refusing to save empty results."
        )

    return payload


def _write_json_atomic(payload: Any, output_file: str | Path) -> None:
    """
    Write JSON atomically to avoid leaving partial/corrupt files.
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(output_path.parent or Path(".")),
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)
        json.dump(payload, tmp, indent=2)
        tmp.write("\n")

    tmp_path.replace(output_path)


def _log_final_result_summary(final_result: Any) -> None:
    """
    Emit useful diagnostics before saving.
    """
    log.info("Final result type before save: %s", type(final_result).__name__)

    if isinstance(final_result, Mapping):
        keys = sorted(str(k) for k in final_result.keys())
        log.info("Final result keys before save: %s", keys)

        log.info("Final target_pdb: %s", final_result.get("target_pdb"))
        log.info(
            "Final binding_results count: %d",
            len(final_result.get("binding_results", []) or []),
        )
        log.info(
            "Final designed_sequences count: %d",
            len(final_result.get("designed_sequences", []) or []),
        )


def main() -> None:
    bootstrap_runtime(__file__)

    parser = argparse.ArgumentParser(description="Virtual Lab Orchestrator")
    parser.add_argument(
        "--topic-file",
        default="research_topics.yaml",
        help="Path to topics YAML",
    )
    parser.add_argument(
        "--topic-index",
        type=int,
        default=None,
        help="Index of topic to run",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum iterations",
    )

    args = parser.parse_args()

    topics = load_topics(args.topic_file)
    topic = select_topic(topics, index=args.topic_index)

    log.info("Starting Virtual Lab for topic: %s", topic["name"])

    bootstrap_knowledge_base(topic)

    wrappers = build_wrapper_bundle()
    data_collector = TrainingDataCollector()

    initial_state = build_initial_state(
        topic=topic,
        wrappers=wrappers,
        max_iterations=args.max_iterations,
        data_collector=data_collector,
    )

    final_result = virtual_lab.invoke(initial_state)

    _log_final_result_summary(final_result)

    payload = _coerce_final_result_for_json(final_result)

    print_user_friendly_summary(final_result)

    output_file = _topic_output_name(topic["name"])
    _write_json_atomic(payload, output_file)

    log.info("Final results saved to %s", output_file)

    try:
        log.info("Running post-run training data pipeline...")
        run_postrun_pipeline(topic, final_result)
    except Exception as e:
        log.warning("Post-run training pipeline failed/non-fatal: %s", e)


if __name__ == "__main__":
    main()
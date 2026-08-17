from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def configure_paths(anchor_file: str | None = None) -> None:
    """
    Ensure VLAB2 and repo root are importable when running scripts directly.
    """
    if anchor_file is None:
        anchor = Path(__file__).resolve()
    else:
        anchor = Path(anchor_file).resolve()

    # VLAB2/orchestration/config.py -> VLAB2 root is parents[1]
    vlab2_root = anchor.parents[1]
    repo_root = anchor.parents[2]

    for path in [str(vlab2_root), str(repo_root)]:
        if path not in sys.path:
            sys.path.insert(0, path)


def configure_environment() -> None:
    """
    Load environment variables and apply safe defaults.
    """
    load_dotenv()

    os.environ.setdefault("HF_HOME", "/mnt/scratch/fbsnpat/bot/VLAB2/hf_cache")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def configure_logging(level: int = logging.INFO) -> None:
    """
    Configure root logging for Virtual Lab runs.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def bootstrap_runtime(anchor_file: str | None = None) -> None:
    """
    Convenience setup for CLI scripts.
    """
    configure_paths(anchor_file)
    configure_environment()
    configure_logging()


def get_peptide_mode() -> str:
    """
    Get peptide design mode from environment variable.
    
    Returns:
        "known_panel" (default, deterministic)
        "llm_design" (uses LLM for generation)
    """
    mode = os.getenv("VLAB_PEPTIDE_MODE", "known_panel").strip().lower()
    
    valid_modes = {"known_panel", "llm_design"}
    if mode not in valid_modes:
        import logging
        log = logging.getLogger("virtual_lab")
        log.warning(
            "Unknown VLAB_PEPTIDE_MODE=%s; using 'known_panel'. "
            "Valid modes: %s",
            mode,
            ", ".join(sorted(valid_modes)),
        )
        return "known_panel"
    
    return mode

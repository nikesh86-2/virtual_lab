from __future__ import annotations

import logging

from VLAB2.core.bioinfo_wrapper import BioinfoWrapper
from VLAB2.core.hdock_wrapper import HDockDocking
from VLAB2.core.md_wrapper import MDWrapper
from VLAB2.core.protein_wrapper import ProteinWrapper
from VLAB2.core.sfold_wrapper import SFoldWrapper
from VLAB2.core.viennarna_wrapper import ViennaRNAWrapper

from dotenv import load_dotenv
from pathlib import Path

PROJECT_ROOT = Path("/mnt/scratch/fbsnpat/bot/VLAB2")
load_dotenv(PROJECT_ROOT / ".env", override=False)

log = logging.getLogger("virtual_lab")


def build_wrapper_bundle() -> dict:
    """
    Build all computational wrappers once at startup.

    Keep this out of agent modules so agents stay pure/state-driven.
    """
    md = MDWrapper()

    try:
        protein = ProteinWrapper(simrna=md)
    except TypeError:
        protein = ProteinWrapper()

    hdock = HDockDocking()

    wrappers = {
        "protein": protein,
        "md": md,
        "sfold": SFoldWrapper(),
        "vienna": ViennaRNAWrapper(),
        "bioinfo": BioinfoWrapper(),
        "hdock": hdock,
    }

    log.info("Wrapper bundle initialised: %s", ", ".join(wrappers.keys()))

    return wrappers
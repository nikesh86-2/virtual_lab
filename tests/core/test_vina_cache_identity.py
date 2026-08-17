from __future__ import annotations

import pytest

from VLAB2.core.vina_wrapper import VinaDocking, VINA_CACHE_SCHEMA_VERSION


def test_vina_cache_key_sensitivity(tmp_path):
    vina = VinaDocking(cache_dir=str(tmp_path))

    rec_file = tmp_path / "rec.pdbqt"
    rec_file.write_text("ATOM 1 N ALA A 1 0.0 0.0 0.0 1.0 0.0 N\nEND")
    lig_file = tmp_path / "lig.pdbqt"
    lig_file.write_text("ATOM 1 C LIG 1 1.0 1.0 1.0 1.0 0.0 C\nEND")

    key1, _ = vina._build_cache_key(
        receptor_pdbqt=str(rec_file),
        ligand_pdbqt=str(lig_file),
        center=(0.0, 0.0, 0.0),
        size=(20.0, 20.0, 20.0),
        exhaustiveness=8,
        seed=1,
        ligand_smiles_sha256="hash_smiles_1",
        ligand_pdbqt_sha256="hash_pdbqt_1",
        manifest_version="2026-07-27-v1",
        record_version=1,
    )

    # Changing SMILES hash must change cache key
    key2, _ = vina._build_cache_key(
        receptor_pdbqt=str(rec_file),
        ligand_pdbqt=str(lig_file),
        center=(0.0, 0.0, 0.0),
        size=(20.0, 20.0, 20.0),
        exhaustiveness=8,
        seed=1,
        ligand_smiles_sha256="hash_smiles_2",
        ligand_pdbqt_sha256="hash_pdbqt_1",
        manifest_version="2026-07-27-v1",
        record_version=1,
    )
    assert key1 != key2

    # Changing manifest version must change cache key
    key3, _ = vina._build_cache_key(
        receptor_pdbqt=str(rec_file),
        ligand_pdbqt=str(lig_file),
        center=(0.0, 0.0, 0.0),
        size=(20.0, 20.0, 20.0),
        exhaustiveness=8,
        seed=1,
        ligand_smiles_sha256="hash_smiles_1",
        ligand_pdbqt_sha256="hash_pdbqt_1",
        manifest_version="2026-07-27-v2",
        record_version=1,
    )
    assert key1 != key3

    # Schema version should be v3
    assert VINA_CACHE_SCHEMA_VERSION == "vina-cache-v3"

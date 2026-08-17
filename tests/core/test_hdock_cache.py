"""
Tests for Priority 5: HDOCK cache portability.

Verifies:
- Cache paths are canonicalised on save
- Cache paths are canonicalised on restore
- Persistent files are validated before cache hit
- Cache roundtrip preserves data correctly
"""

import json
import os
import pytest
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestHDOCKCachePortability:
    """Test HDOCK cache portability."""

    def test_cache_save_canonicalises_paths(self, tmp_path):
        """Cache save should canonicalise paths."""
        from core.hdock_wrapper import HDockDocking

        wrapper = HDockDocking()

        # Create mock result with relative paths
        result = {
            "output_file": "relative/path/output.out",
            "dock_output_file": "relative/path/output.out",
            "complex_file": "relative/path/complex.out",
            "dock_complex_file": "relative/path/complex.out",
            "complex_status": {
                "output_file": "relative/path/complex.out",
                "valid": True,
            },
        }

        # Call the internal save method
        cache_key = "test_key_123"
        output_out = str(tmp_path / "output.out")
        output_complex = str(tmp_path / "complex.out")

        # Create actual files
        Path("relative/path").mkdir(parents=True, exist_ok=True)
        Path("relative/path/output.out").write_text("output data")
        Path("relative/path/complex.out").write_text("complex data")

        wrapper._save_cached_result(
            cache_key=cache_key,
            result=result,
            output_out=output_out,
            output_complex=output_complex,
            cache_payload={"test": "data"},
        )

        # Check that cached files exist
        metadata_path, cached_output, cached_complex = wrapper._cache_paths(cache_key)
        assert metadata_path.exists()

        # Load and verify the cached result has canonical paths
        with open(metadata_path) as f:
            metadata = json.load(f)

        cached_result = metadata["result"]
        # Paths should point to cached locations, not relative paths
        assert cached_result["output_file"] == str(cached_output)
        assert cached_result["dock_output_file"] == str(cached_output)

        # Cleanup
        shutil.rmtree("relative", ignore_errors=True)

    def test_cache_restore_canonicalises_paths(self, tmp_path):
        """Cache restore should canonicalise paths."""
        from core.hdock_wrapper import HDockDocking

        wrapper = HDockDocking()

        # First save a result
        result = {
            "valid": True,  # Required by _load_cached_result
            "output_file": str(tmp_path / "output.out"),
            "dock_output_file": str(tmp_path / "output.out"),
            "complex_file": str(tmp_path / "complex.out"),
            "dock_complex_file": str(tmp_path / "complex.out"),
            "complex_status": {
                "output_file": str(tmp_path / "complex.out"),
                "valid": True,
            },
            "score": -100.0,
        }

        output_out = str(tmp_path / "output.out")
        output_complex = str(tmp_path / "complex.out")

        # Create actual files
        Path(output_out).parent.mkdir(parents=True, exist_ok=True)
        Path(output_out).write_text("output data")
        Path(output_complex).write_text("complex data")

        cache_key = "test_key_456"
        wrapper._save_cached_result(
            cache_key=cache_key,
            result=result,
            output_out=output_out,
            output_complex=output_complex,
            cache_payload={"test": "data"},
        )

        # Now restore to different paths
        restore_out = str(tmp_path / "restore" / "output.out")
        restore_complex = str(tmp_path / "restore" / "complex.out")

        restored = wrapper._load_cached_result(
            cache_key=cache_key,
            output_out=restore_out,
            output_complex=restore_complex,
        )

        # Verify restored paths are canonical
        assert restored is not None
        assert restored["output_file"] == restore_out
        assert restored["dock_output_file"] == restore_out
        assert restored["complex_file"] == restore_complex
        assert restored["dock_complex_file"] == restore_complex

    def test_no_temp_paths_in_restored_result(self, tmp_path):
        """Restored result should not contain temp paths."""
        from core.hdock_wrapper import HDockDocking

        wrapper = HDockDocking()

        # Save with temp-like paths
        result = {
            "valid": True,  # Required by _load_cached_result
            "output_file": "/tmp/hdock_tmp_123.out",
            "dock_output_file": "/tmp/hdock_tmp_123.out",
            "complex_file": "/tmp/hdock_tmp_456.out",
            "dock_complex_file": "/tmp/hdock_tmp_456.out",
            "complex_status": {
                "output_file": "/tmp/hdock_tmp_456.out",
                "valid": True,
            },
        }

        output_out = str(tmp_path / "output.out")
        output_complex = str(tmp_path / "complex.out")

        # Create actual files
        Path(output_out).write_text("output data")
        Path(output_complex).write_text("complex data")

        cache_key = "test_key_789"
        wrapper._save_cached_result(
            cache_key=cache_key,
            result=result,
            output_out=output_out,
            output_complex=output_complex,
            cache_payload={"test": "data"},
        )

        # Restore to new location
        restore_out = str(tmp_path / "restore" / "output.out")
        restore_complex = str(tmp_path / "restore" / "complex.out")

        restored = wrapper._load_cached_result(
            cache_key=cache_key,
            output_out=restore_out,
            output_complex=restore_complex,
        )

        # Verify no temp paths in result
        assert restored is not None
        result_str = json.dumps(restored)
        assert "/tmp/hdock_tmp" not in result_str
        assert ".tmp" not in result_str

    def test_deleted_persistent_file_causes_cache_miss(self, tmp_path):
        """Deleted persistent file should cause cache miss."""
        from core.hdock_wrapper import HDockDocking

        wrapper = HDockDocking()

        # Save a result
        result = {
            "output_file": str(tmp_path / "output.out"),
            "dock_output_file": str(tmp_path / "output.out"),
            "complex_file": str(tmp_path / "complex.out"),
            "dock_complex_file": str(tmp_path / "complex.out"),
        }

        output_out = str(tmp_path / "output.out")
        output_complex = str(tmp_path / "complex.out")

        Path(output_out).write_text("output data")
        Path(output_complex).write_text("complex data")

        cache_key = "test_key_del"
        wrapper._save_cached_result(
            cache_key=cache_key,
            result=result,
            output_out=output_out,
            output_complex=output_complex,
            cache_payload={"test": "data"},
        )

        # Delete the cached output file
        metadata_path, cached_output, cached_complex = wrapper._cache_paths(cache_key)
        cached_output.unlink()

        # Restore should fail (cache miss)
        restore_out = str(tmp_path / "restore" / "output.out")
        restore_complex = str(tmp_path / "restore" / "complex.out")

        restored = wrapper._load_cached_result(
            cache_key=cache_key,
            output_out=restore_out,
            output_complex=restore_complex,
        )

        assert restored is None

    def test_empty_persistent_file_causes_cache_miss(self, tmp_path):
        """Empty persistent file should cause cache miss."""
        from core.hdock_wrapper import HDockDocking

        wrapper = HDockDocking()

        # Save a result
        result = {
            "output_file": str(tmp_path / "output.out"),
            "dock_output_file": str(tmp_path / "output.out"),
            "complex_file": str(tmp_path / "complex.out"),
            "dock_complex_file": str(tmp_path / "complex.out"),
        }

        output_out = str(tmp_path / "output.out")
        output_complex = str(tmp_path / "complex.out")

        Path(output_out).write_text("output data")
        Path(output_complex).write_text("complex data")

        cache_key = "test_key_empty"
        wrapper._save_cached_result(
            cache_key=cache_key,
            result=result,
            output_out=output_out,
            output_complex=output_complex,
            cache_payload={"test": "data"},
        )

        # Make cached output file empty
        metadata_path, cached_output, cached_complex = wrapper._cache_paths(cache_key)
        cached_output.write_text("")

        # Restore should fail (cache miss)
        restore_out = str(tmp_path / "restore" / "output.out")
        restore_complex = str(tmp_path / "restore" / "complex.out")

        restored = wrapper._load_cached_result(
            cache_key=cache_key,
            output_out=restore_out,
            output_complex=restore_complex,
        )

        assert restored is None

    def test_valid_persistent_files_pass_validation(self, tmp_path):
        """Valid persistent files should pass validation."""
        from core.hdock_wrapper import HDockDocking

        wrapper = HDockDocking()

        # Save a result
        result = {
            "valid": True,  # Required by _load_cached_result
            "output_file": str(tmp_path / "output.out"),
            "dock_output_file": str(tmp_path / "output.out"),
            "complex_file": str(tmp_path / "complex.out"),
            "dock_complex_file": str(tmp_path / "complex.out"),
            "score": -100.0,
        }

        output_out = str(tmp_path / "output.out")
        output_complex = str(tmp_path / "complex.out")

        Path(output_out).write_text("output data")
        Path(output_complex).write_text("complex data")

        cache_key = "test_key_valid"
        wrapper._save_cached_result(
            cache_key=cache_key,
            result=result,
            output_out=output_out,
            output_complex=output_complex,
            cache_payload={"test": "data"},
        )

        # Restore should succeed
        restore_out = str(tmp_path / "restore" / "output.out")
        restore_complex = str(tmp_path / "restore" / "complex.out")

        restored = wrapper._load_cached_result(
            cache_key=cache_key,
            output_out=restore_out,
            output_complex=restore_complex,
        )

        assert restored is not None
        assert restored["score"] == -100.0


class TestHDOCKCacheIntegration:
    """Integration tests for HDOCK cache."""

    def test_cache_roundtrip_preserves_data(self, tmp_path):
        """Cache roundtrip should preserve all data."""
        from core.hdock_wrapper import HDockDocking

        wrapper = HDockDocking()

        # Original result with all fields
        result = {
            "valid": True,  # Required by _load_cached_result
            "output_file": str(tmp_path / "original" / "output.out"),
            "dock_output_file": str(tmp_path / "original" / "output.out"),
            "complex_file": str(tmp_path / "original" / "complex.out"),
            "dock_complex_file": str(tmp_path / "original" / "complex.out"),
            "complex_status": {
                "output_file": str(tmp_path / "original" / "complex.out"),
                "valid": True,
            },
            "score": -150.5,
            "energy": -200.0,
            "rmsd": 5.5,
            "num_models": 10,
        }

        output_out = str(tmp_path / "original" / "output.out")
        output_complex = str(tmp_path / "original" / "complex.out")

        Path(output_out).parent.mkdir(parents=True, exist_ok=True)
        Path(output_out).write_text("output content here")
        Path(output_complex).write_text("complex content here")

        cache_key = "test_roundtrip"
        wrapper._save_cached_result(
            cache_key=cache_key,
            result=result,
            output_out=output_out,
            output_complex=output_complex,
            cache_payload={"receptor": "test.pdb", "ligand": "test_rna.pdb"},
        )

        # Restore to new location
        restore_out = str(tmp_path / "restored" / "output.out")
        restore_complex = str(tmp_path / "restored" / "complex.out")

        restored = wrapper._load_cached_result(
            cache_key=cache_key,
            output_out=restore_out,
            output_complex=restore_complex,
        )

        # Verify all data preserved
        assert restored is not None
        assert restored["score"] == -150.5
        assert restored["energy"] == -200.0
        assert restored["rmsd"] == 5.5
        assert restored["num_models"] == 10
        assert restored["cache_hit"] is True

        # Verify files were copied
        assert Path(restore_out).exists()
        assert Path(restore_out).read_text() == "output content here"

    def test_cache_metadata_file_recorded(self, tmp_path):
        """Cache metadata file should be recorded in restored result."""
        from core.hdock_wrapper import HDockDocking

        wrapper = HDockDocking()

        result = {
            "valid": True,  # Required by _load_cached_result
            "output_file": str(tmp_path / "output.out"),
            "dock_output_file": str(tmp_path / "output.out"),
            "complex_file": str(tmp_path / "complex.out"),
            "dock_complex_file": str(tmp_path / "complex.out"),
        }

        output_out = str(tmp_path / "output.out")
        output_complex = str(tmp_path / "complex.out")

        Path(output_out).write_text("output")
        Path(output_complex).write_text("complex")

        cache_key = "test_metadata"
        wrapper._save_cached_result(
            cache_key=cache_key,
            result=result,
            output_out=output_out,
            output_complex=output_complex,
            cache_payload={"test": "data"},
        )

        # Get metadata path
        metadata_path, _, _ = wrapper._cache_paths(cache_key)

        # Restore
        restore_out = str(tmp_path / "restore" / "output.out")
        restore_complex = str(tmp_path / "restore" / "complex.out")

        restored = wrapper._load_cached_result(
            cache_key=cache_key,
            output_out=restore_out,
            output_complex=restore_complex,
        )

        # Verify metadata file is recorded
        assert restored is not None
        assert "cache_metadata_file" in restored
        assert restored["cache_metadata_file"] == str(metadata_path)
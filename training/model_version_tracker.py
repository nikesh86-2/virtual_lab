"""
model_version_tracker.py

Track model versions, maintain history, cleanup old models, and provide rollback capability.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger("virtual_lab.training")

VERSIONS_DIR = Path("training/merged_model_versions")
ACTIVE_MODEL_FILE = Path("training/active_model.json")


def setup_versioning() -> None:
    """Setup versioning directory structure."""
    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)


def get_active_model() -> Optional[dict]:
    """Get the currently active model information."""
    if not ACTIVE_MODEL_FILE.exists():
        return None
    
    try:
        with open(ACTIVE_MODEL_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log.warning(f"Failed to read active model file: {e}")
        return None


def set_active_model(model_path: str, version_id: str, metadata: Optional[dict] = None) -> None:
    """Set the active model."""
    active_info = {
        "version_id": version_id,
        "model_path": str(model_path),
        "activated_at": datetime.now().isoformat(),
        "metadata": metadata or {},
    }
    
    with open(ACTIVE_MODEL_FILE, "w", encoding="utf-8") as f:
        json.dump(active_info, f, indent=2)
    
    log.info(f"Active model set to {version_id} at {model_path}")


def register_model_version(model_path: str, metadata: Optional[dict] = None) -> str:
    """Register a new model version."""
    setup_versioning()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_id = f"v{timestamp}"
    
    version_dir = VERSIONS_DIR / version_id
    
    # Copy model to version directory
    if Path(model_path).exists():
        if version_dir.exists():
            shutil.rmtree(version_dir)
        shutil.copytree(model_path, version_dir)
        log.info(f"Registered model version {version_id} at {version_dir}")
    else:
        log.warning(f"Model path {model_path} does not exist, creating empty version entry")
        version_dir.mkdir(parents=True, exist_ok=True)
    
    # Save version metadata
    version_info = {
        "version_id": version_id,
        "model_path": str(version_dir),
        "created_at": datetime.now().isoformat(),
        "metadata": metadata or {},
    }
    
    version_file = version_dir / "version_info.json"
    with open(version_file, "w", encoding="utf-8") as f:
        json.dump(version_info, f, indent=2)
    
    return version_id


def list_model_versions() -> list[dict]:
    """List all available model versions."""
    setup_versioning()
    
    versions = []
    
    for version_dir in sorted(VERSIONS_DIR.iterdir()):
        if not version_dir.is_dir():
            continue
        
        version_file = version_dir / "version_info.json"
        if version_file.exists():
            try:
                with open(version_file, "r", encoding="utf-8") as f:
                    version_info = json.load(f)
                versions.append(version_info)
            except Exception as e:
                log.warning(f"Failed to read version info for {version_dir}: {e}")
    
    return versions


def cleanup_old_versions(max_versions: int = 3) -> None:
    """Cleanup old model versions, keeping only the most recent max_versions."""
    versions = list_model_versions()
    
    if len(versions) <= max_versions:
        log.info(f"Only {len(versions)} versions, no cleanup needed (max: {max_versions})")
        return
    
    # Get active model to avoid deleting it
    active_model = get_active_model()
    active_version_id = active_model.get("version_id") if active_model else None
    
    # Sort by creation time (newest first)
    versions.sort(key=lambda x: x["created_at"], reverse=True)
    
    # Keep max_versions versions, excluding active version
    to_delete = []
    kept = 0
    
    for version in versions:
        if version["version_id"] == active_version_id:
            # Always keep active version
            continue
        
        if kept < max_versions:
            kept += 1
        else:
            to_delete.append(version)
    
    # Delete old versions
    for version in to_delete:
        version_path = Path(version["model_path"])
        if version_path.exists():
            try:
                shutil.rmtree(version_path)
                log.info(f"Deleted old model version {version['version_id']}")
            except Exception as e:
                log.warning(f"Failed to delete version {version['version_id']}: {e}")


def rollback_to_version(version_id: str) -> bool:
    """Rollback to a specific model version."""
    versions = list_model_versions()
    
    # Find the requested version
    target_version = None
    for version in versions:
        if version["version_id"] == version_id:
            target_version = version
            break
    
    if not target_version:
        log.error(f"Version {version_id} not found")
        return False
    
    model_path = target_version["model_path"]
    if not Path(model_path).exists():
        log.error(f"Model path {model_path} does not exist")
        return False
    
    # Set as active model
    set_active_model(
        model_path=model_path,
        version_id=version_id,
        metadata=target_version.get("metadata"),
    )
    
    log.info(f"Rolled back to version {version_id}")
    return True


def get_latest_version() -> Optional[dict]:
    """Get the most recent model version."""
    versions = list_model_versions()
    
    if not versions:
        return None
    
    # Sort by creation time (newest first)
    versions.sort(key=lambda x: x["created_at"], reverse=True)
    return versions[0]


def main() -> None:
    """CLI interface for model version management."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python model_version_tracker.py <command> [args]")
        print("Commands:")
        print("  list - List all model versions")
        print("  register <model_path> - Register a new model version")
        print("  activate <version_id> - Activate a specific version")
        print("  rollback <version_id> - Rollback to a specific version")
        print("  cleanup [max_versions] - Cleanup old versions (default: 3)")
        print("  active - Show currently active model")
        return
    
    command = sys.argv[1]
    
    if command == "list":
        versions = list_model_versions()
        print(f"Found {len(versions)} model versions:")
        for version in versions:
            active_marker = " (ACTIVE)" if get_active_model() and get_active_model().get("version_id") == version["version_id"] else ""
            print(f"  {version['version_id']}: {version['model_path']}{active_marker}")
            print(f"    Created: {version['created_at']}")
    
    elif command == "register":
        if len(sys.argv) < 3:
            print("Usage: python model_version_tracker.py register <model_path>")
            return
        model_path = sys.argv[2]
        version_id = register_model_version(model_path)
        print(f"Registered version {version_id}")
    
    elif command == "activate":
        if len(sys.argv) < 3:
            print("Usage: python model_version_tracker.py activate <version_id>")
            return
        version_id = sys.argv[2]
        versions = list_model_versions()
        target = None
        for v in versions:
            if v["version_id"] == version_id:
                target = v
                break
        if target:
            set_active_model(target["model_path"], version_id, target.get("metadata"))
            print(f"Activated version {version_id}")
        else:
            print(f"Version {version_id} not found")
    
    elif command == "rollback":
        if len(sys.argv) < 3:
            print("Usage: python model_version_tracker.py rollback <version_id>")
            return
        version_id = sys.argv[2]
        if rollback_to_version(version_id):
            print(f"Rolled back to version {version_id}")
        else:
            print(f"Failed to rollback to version {version_id}")
    
    elif command == "cleanup":
        max_versions = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        cleanup_old_versions(max_versions)
        print(f"Cleanup complete (keeping max {max_versions} versions)")
    
    elif command == "active":
        active = get_active_model()
        if active:
            print(f"Active model: {active['version_id']}")
            print(f"Path: {active['model_path']}")
            print(f"Activated at: {active['activated_at']}")
        else:
            print("No active model set")
    
    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()

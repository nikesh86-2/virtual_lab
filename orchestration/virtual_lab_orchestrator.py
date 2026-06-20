from __future__ import annotations

import sys
from pathlib import Path


# This file lives at:
#   <repo_root>/VLAB2/orchestration/virtual_lab_orchestrator.py
#
# We need <repo_root> on sys.path so imports like
#   from VLAB2.orchestration.cli import main
# work when this script is run directly.
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from VLAB2.orchestration.cli import main


if __name__ == "__main__":
    main()

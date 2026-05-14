"""Tiny launcher for the OpenReview supplement's federatedscope/main.py.

Forwards argv to main.py, but first monkey-patches socket.gethostbyname so the
supplement's logging.py doesn't crash on macOS hosts whose hostname isn't in
/etc/hosts (FederatedScope unconditionally calls gethostbyname(gethostname())).

Usage:
    SLS_ALTERNATION_MODE=rolora python scripts/run_supplement.py --cfg <yaml>
"""

from __future__ import annotations

import socket
import sys
from pathlib import Path

_orig_gethostbyname = socket.gethostbyname


def _safe_gethostbyname(host: str) -> str:
    try:
        return _orig_gethostbyname(host)
    except socket.gaierror:
        return "127.0.0.1"


socket.gethostbyname = _safe_gethostbyname  # type: ignore[assignment]

REPO = Path(__file__).resolve().parents[1]
SUPP = REPO / "code" / "harness" / "rolora-supplement" / "RoLoRA-code"
sys.path.insert(0, str(SUPP))

if __name__ == "__main__":
    # Re-exec the supplement's main.py inside this interpreter so the monkey-patch
    # is live for everything it imports.
    main_path = SUPP / "federatedscope" / "main.py"
    with open(main_path, encoding="utf-8") as fh:
        code = fh.read()
    sys.argv[0] = str(main_path)
    exec(compile(code, str(main_path), "exec"), {"__name__": "__main__", "__file__": str(main_path)})

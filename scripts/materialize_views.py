#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from discernmem.config import DiscernMemConfig
from discernmem.storage import DiscernMemStore


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize DiscernMem reports from authoritative records without "
            "changing memory state."
        )
    )
    parser.add_argument("--storage-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--complete",
        action="store_true",
        help="Mark generated reports as a final, complete experiment snapshot.",
    )
    args = parser.parse_args()

    store = DiscernMemStore(
        args.storage_dir,
        DiscernMemConfig.from_yaml(args.config),
    )
    summary = store.materialize_views(complete=args.complete)
    output = {
        "records_dir": str(store.records_root),
        "reports_dir": str(store.reports_root),
        "memory_index": str(store.report_index_path),
        "full_export": str(store.full_export_path),
        "summary": summary,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

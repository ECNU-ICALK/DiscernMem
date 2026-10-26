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
from discernmem.metrics import DiscernMemMetrics
from discernmem.storage import DiscernMemStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute DiscernMem metrics from inspectable JSON artifacts.")
    parser.add_argument("--storage-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    store = DiscernMemStore(args.storage_dir, DiscernMemConfig.from_yaml(args.config))
    output = DiscernMemMetrics(store).write(args.output)
    metrics = json.loads(output.read_text(encoding="utf-8"))
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if metrics["final_test_leakage"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dermbench.scoring import METRICS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize DermBench score JSONL files.")
    parser.add_argument("--scores", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    values = {metric: [] for metric in METRICS}
    with Path(args.scores).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            for metric, score in row.get("scores", {}).items():
                if metric in values:
                    values[metric].append(float(score))

    summary = {
        metric: round(sum(scores) / len(scores), 3) if scores else None
        for metric, scores in values.items()
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

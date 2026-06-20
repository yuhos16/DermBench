#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dermbench.data import load_manifest, sha256_file
from dermbench.scoring import METRICS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the DermBench release files.")
    parser.add_argument("--manifest", default="data/manifest/dermbench_test.jsonl")
    parser.add_argument("--expected-count", type=int, default=4000)
    parser.add_argument("--expected-categories", type=int, default=23)
    parser.add_argument("--check-sha", action="store_true")
    parser.add_argument("--image-root", default=None, help="Optional local DermNet test image root.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = Path(args.manifest)
    cases = load_manifest(manifest)
    errors: list[str] = []

    if len(cases) != args.expected_count:
        errors.append(f"expected {args.expected_count} cases, found {len(cases)}")

    counts = Counter(case.category for case in cases)
    if len(counts) != args.expected_categories:
        errors.append(f"expected {args.expected_categories} categories, found {len(counts)}")

    seen_ids: set[str] = set()
    for case in cases:
        if case.id in seen_ids:
            errors.append(f"duplicate id: {case.id}")
        seen_ids.add(case.id)
        if not case.cot.exists():
            errors.append(f"missing CoT text: {case.cot}")
            continue
        if not case.diagnosis:
            errors.append(f"missing diagnosis: {case.id}")
        if args.check_sha and sha256_file(case.cot) != case.cot_sha256:
            errors.append(f"CoT sha mismatch: {case.id}")
        if args.image_root and not case.image_file(args.image_root).exists():
            errors.append(f"missing external DermNet image: {case.image_file(args.image_root)}")

    config_path = Path("dermbench/configs/models.json")
    prompt_paths = [
        Path("dermbench/prompts/candidate_generation.txt"),
        Path("dermbench/prompts/dermbench_judge.txt"),
    ]
    for path in prompt_paths + [config_path]:
        if not path.exists():
            errors.append(f"missing required file: {path}")

    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        configured_metrics = config.get("metrics", [])
        if configured_metrics != METRICS:
            errors.append("dermbench/configs/models.json metric order does not match dermbench.scoring.METRICS")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "cases": len(cases),
                "categories": len(counts),
                "bundled_images": False,
                "metrics": METRICS,
                "status": "ok",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

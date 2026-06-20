#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openai import OpenAI
from tqdm import tqdm

from dermbench.data import iter_limited, load_manifest
from dermbench.scoring import load_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate DermBench candidate narratives with OpenAI.")
    parser.add_argument("--manifest", default="data/manifest/dermbench_test.jsonl")
    parser.add_argument("--image-root", required=True, help="Local root of the DermNet test images.")
    parser.add_argument("--prompt", default="prompts/candidate_generation.txt")
    parser.add_argument("--output", default="outputs/candidates.jsonl")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between requests.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required.")

    cases = list(iter_limited(load_manifest(args.manifest), args.limit))
    prompt = load_prompt(args.prompt)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    client = OpenAI()

    with output_path.open("a", encoding="utf-8") as handle:
        for case in tqdm(cases, desc="Generating"):
            response = client.responses.create(
                model=args.model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": case.image_data_url(args.image_root)},
                        ],
                    }
                ],
            )
            row = {
                "id": case.id,
                "model": args.model,
                "candidate_text": (response.output_text or "").strip(),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            if args.sleep:
                time.sleep(args.sleep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

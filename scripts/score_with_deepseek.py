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

from dermbench.data import load_manifest
from dermbench.scoring import METRICS, load_prompt, parse_scores, render_judge_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score DermBench candidate narratives with DeepSeek-R1.")
    parser.add_argument("--manifest", default="data/manifest/dermbench_test.jsonl")
    parser.add_argument("--candidates", required=True, help="JSONL with id and candidate_text fields.")
    parser.add_argument("--prompt", default="prompts/dermbench_judge.txt")
    parser.add_argument("--output", default="outputs/dermbench_scores.jsonl")
    parser.add_argument("--model", default=os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner"))
    parser.add_argument("--base-url", default=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between requests.")
    return parser.parse_args()


def read_candidates(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> int:
    args = parse_args()
    if not os.getenv("DEEPSEEK_API_KEY"):
        raise RuntimeError("DEEPSEEK_API_KEY is required.")

    cases = {case.id: case for case in load_manifest(args.manifest)}
    prompt_template = load_prompt(args.prompt)
    client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url=args.base_url)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("a", encoding="utf-8") as handle:
        for candidate in tqdm(read_candidates(args.candidates), desc="Scoring"):
            case_id = candidate["id"]
            case = cases[case_id]
            candidate_text = candidate.get("candidate_text") or candidate.get("text") or ""
            judge_prompt = render_judge_prompt(
                prompt_template,
                candidate_text=candidate_text,
                reference_text=case.load_reference(),
            )
            response = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": judge_prompt}],
            )
            judge_text = response.choices[0].message.content or ""
            scores = parse_scores(judge_text)
            missing = [metric for metric in METRICS if metric not in scores]
            row = {
                "id": case_id,
                "candidate_model": candidate.get("model", ""),
                "judge_model": args.model,
                "scores": scores,
                "missing_metrics": missing,
                "judge_text": judge_text.strip(),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            if args.sleep:
                time.sleep(args.sleep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

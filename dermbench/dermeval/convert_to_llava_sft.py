"""Convert DermEval raw JSON/JSONL data into LLaVA supervised fine-tuning JSON.

Raw record schema:
{
  "id": "case_001",
  "image": "relative/path.jpg",
  "diagnostic_text": "candidate narrative to be evaluated",
  "evaluation_text": "Accuracy: 4.0 - ...\nSafety: 5.0 - ...",   # optional
  "scores": {"accuracy": 4, "safety": 5, ...},                  # required if no evaluation_text
  "rationales": {"accuracy": "..."}                              # optional
}
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List

from .dataio import get_diagnostic_text, get_evaluation_text, load_json_or_jsonl, validate_dermeval_record, write_json
from .prompts import build_llava_conversations


def convert_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx, rec in enumerate(records):
        validate_dermeval_record(rec, require_scores=("evaluation_text" not in rec and "label_text" not in rec))
        diagnostic_text = get_diagnostic_text(rec)
        evaluation_text = get_evaluation_text(rec)
        out.append(
            {
                "id": rec.get("id", str(idx)),
                "image": rec["image"],
                "conversations": build_llava_conversations(diagnostic_text, evaluation_text),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert DermEval raw records to LLaVA SFT format.")
    parser.add_argument("--input", required=True, help="Raw DermEval JSON/JSONL.")
    parser.add_argument("--output", required=True, help="Output LLaVA SFT JSON path.")
    args = parser.parse_args()
    records = load_json_or_jsonl(args.input)
    write_json(args.output, convert_records(records))
    print(f"Wrote {len(records)} LLaVA SFT samples to {args.output}")


if __name__ == "__main__":
    main()

"""Input/output helpers for DermEval raw and LLaVA-style training data."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

from .constants import DERMEVAL_DIMENSION_KEYS
from .prompts import canonical_evaluation_text
from .score_parser import normalize_target_scores


def load_json_or_jsonl(path: str | os.PathLike[str]) -> List[Dict[str, Any]]:
    """Load a JSON list or JSONL file."""
    path = str(path)
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        data = json.load(f)
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    if not isinstance(data, list):
        raise ValueError("Input must be a JSON list, a {'data': [...]} object, or JSONL.")
    return data


def write_json(path: str | os.PathLike[str], data: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_jsonl(path: str | os.PathLike[str], rows: Iterable[Mapping[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def get_scores(record: Mapping[str, Any]) -> List[float]:
    """Read a score vector from a DermEval raw record.

    Accepted forms:
      - ``scores`` dict with canonical keys;
      - ``scores`` six-element list in DermEval order;
      - six top-level canonical score fields.
    """
    if "scores" in record:
        return normalize_target_scores(record["scores"])
    if all(key in record for key in DERMEVAL_DIMENSION_KEYS):
        return normalize_target_scores({key: record[key] for key in DERMEVAL_DIMENSION_KEYS})
    raise KeyError(
        "Record must contain a 'scores' dict/list or top-level score fields: "
        + ", ".join(DERMEVAL_DIMENSION_KEYS)
    )


def get_diagnostic_text(record: Mapping[str, Any]) -> str:
    for key in ("diagnostic_text", "candidate_text", "narrative", "answer", "text"):
        if key in record and str(record[key]).strip():
            return str(record[key]).strip()
    raise KeyError("Record must contain diagnostic_text/candidate_text/narrative/answer/text.")


def get_evaluation_text(record: Mapping[str, Any]) -> str:
    """Return evaluation label text, or construct one from scores/rationales."""
    for key in ("evaluation_text", "label_text", "critique", "rationale"):
        if key in record and str(record[key]).strip():
            return str(record[key]).strip()
    scores = get_scores(record)
    rationales = record.get("rationales") or record.get("justifications") or {}
    return canonical_evaluation_text(scores, rationales=rationales)


def validate_dermeval_record(record: Mapping[str, Any], require_scores: bool = True) -> None:
    if "image" not in record:
        raise KeyError("Record must contain an 'image' relative path.")
    _ = get_diagnostic_text(record)
    if require_scores:
        _ = get_scores(record)

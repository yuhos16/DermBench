from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


METRICS = [
    "Accuracy",
    "Safety",
    "Medical Groundedness",
    "Clinical Coverage",
    "Reasoning Coherence",
    "Description Precision",
]


def load_prompt(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def render_judge_prompt(template: str, candidate_text: str, reference_text: str) -> str:
    return template.format(
        candidate_text=candidate_text.strip(),
        reference_text=reference_text.strip(),
    )


def parse_scores(text: str) -> dict[str, float]:
    """Extract six DermBench scores from JSON or simple metric lines."""
    scores = _parse_json_scores(text)
    if scores:
        return scores

    parsed: dict[str, float] = {}
    for metric in METRICS:
        pattern = rf"{re.escape(metric)}\s*:\s*([0-5](?:\.\d+)?)"
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            parsed[metric] = _clamp_score(float(match.group(1)))
    return parsed


def _parse_json_scores(text: str) -> dict[str, float]:
    candidates = [text]
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.insert(0, fenced.group(1))

    for candidate in candidates:
        try:
            data: Any = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        raw_scores = data.get("scores", data)
        if not isinstance(raw_scores, dict):
            continue
        parsed = {}
        for metric in METRICS:
            value = raw_scores.get(metric)
            if value is None:
                continue
            try:
                parsed[metric] = _clamp_score(float(value))
            except (TypeError, ValueError):
                continue
        if parsed:
            return parsed
    return {}


def _clamp_score(value: float) -> float:
    return max(0.0, min(5.0, value))


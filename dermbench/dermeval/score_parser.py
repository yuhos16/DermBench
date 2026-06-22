"""Score parsing and reward functions for DermEval.

The paper uses an external LLM parser to extract scores. For a reproducible open
implementation, this file provides a deterministic regex parser that understands
common variants such as "4/5", "8/10", and alternate dimension names.
"""

from __future__ import annotations

import argparse
import json
import re
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .constants import DEFAULT_SCORE_SCALE, DERMEVAL_DIMENSIONS, DERMEVAL_DIMENSION_KEYS, DIMENSION_ALIASES

_NUMBER = r"([-+]?\d+(?:\.\d+)?)"
_DENOM = r"(?:\s*/\s*(\d+(?:\.\d+)?))?"


def _clip(value: float, low: float = 0.0, high: float = DEFAULT_SCORE_SCALE) -> float:
    return max(low, min(high, float(value)))


def _normalize_score(value: float, denom: Optional[float], target_scale: float = DEFAULT_SCORE_SCALE) -> float:
    value = float(value)
    if denom and denom > 0:
        value = value / denom * target_scale
    elif value > target_scale and value <= 10.0:
        # Common evaluator output is 0--10. Convert it to 0--5 if no denominator is shown.
        value = value / 10.0 * target_scale
    return _clip(value, 0.0, target_scale)


def parse_scores(text: str, target_scale: float = DEFAULT_SCORE_SCALE) -> Dict[str, Optional[float]]:
    """Extract six DermEval scores from free-form text.

    Returns a dict keyed by canonical dimension names. Missing dimensions are
    represented by None. Scores are converted to a 0--5 scale and clipped.
    """
    text = str(text or "")
    scores: Dict[str, Optional[float]] = {key: None for key in DERMEVAL_DIMENSION_KEYS}

    for key, aliases in DIMENSION_ALIASES.items():
        alias_pattern = "|".join(re.escape(alias) for alias in sorted(aliases, key=len, reverse=True))
        patterns = [
            rf"(?im)(?:^|\n|\b)(?:{alias_pattern})\s*(?:score|rating)?\s*[:=\-]\s*{_NUMBER}{_DENOM}",
            rf"(?im)(?:^|\n|\b)(?:{alias_pattern})\s*\(?\s*{_NUMBER}{_DENOM}\s*\)?",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if not match:
                continue
            value = float(match.group(1))
            denom = float(match.group(2)) if len(match.groups()) > 1 and match.group(2) else None
            scores[key] = _normalize_score(value, denom, target_scale=target_scale)
            break
    return scores


def parsed_scores_to_list(parsed: Mapping[str, Optional[float]]) -> List[Optional[float]]:
    return [parsed.get(key) for key in DERMEVAL_DIMENSION_KEYS]


def normalize_target_scores(scores: Mapping[str, float] | Sequence[float]) -> List[float]:
    if isinstance(scores, Mapping):
        return [_clip(float(scores[key])) for key in DERMEVAL_DIMENSION_KEYS]
    values = [float(x) for x in scores]
    if len(values) != len(DERMEVAL_DIMENSION_KEYS):
        raise ValueError(f"Expected {len(DERMEVAL_DIMENSION_KEYS)} scores, got {len(values)}")
    return [_clip(v) for v in values]


def reward_negative_mse(
    predicted: Mapping[str, Optional[float]] | Sequence[Optional[float]],
    target: Mapping[str, float] | Sequence[float],
    missing_reward: Optional[float] = None,
) -> Tuple[float, int]:
    """Compute the SOREB reward: negative MSE over successfully parsed dimensions.

    Args:
        predicted: Parsed scores keyed by dimension or in DermEval order.
        target: Physician scores keyed by dimension or in DermEval order.
        missing_reward: Reward to use if no score can be parsed. If None, the
            function returns ``(nan, 0)``.

    Returns:
        (reward, number_of_valid_dimensions)
    """
    if isinstance(predicted, Mapping):
        pred_list = parsed_scores_to_list(predicted)
    else:
        pred_list = list(predicted)
    tgt_list = normalize_target_scores(target)
    valid = [(float(p), float(t)) for p, t in zip(pred_list, tgt_list) if p is not None]
    if not valid:
        if missing_reward is None:
            return float("nan"), 0
        return float(missing_reward), 0
    mse = sum((p - t) ** 2 for p, t in valid) / len(valid)
    return -float(mse), len(valid)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse DermEval scores from text.")
    parser.add_argument("text_file", help="A text file containing a DermEval output.")
    args = parser.parse_args()
    with open(args.text_file, "r", encoding="utf-8") as f:
        text = f.read()
    print(json.dumps(parse_scores(text), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

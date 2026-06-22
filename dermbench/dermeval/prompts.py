"""Prompt and output templates for DermEval.

The functions here convert a candidate diagnostic narrative into the exact LLaVA
conversation format used in supervised fine-tuning and SOREB training.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence

try:
    from llava.constants import DEFAULT_IMAGE_TOKEN
except Exception:  # Allows data conversion without importing the full LLaVA stack.
    DEFAULT_IMAGE_TOKEN = "<image>"

from .constants import DERMEVAL_DIMENSIONS, DERMEVAL_DIMENSION_KEYS, RUBRIC_SHORT


def _clean_text(text: str) -> str:
    return " ".join(str(text).replace("\r", "\n").split())


def build_dermeval_prompt(diagnostic_text: str) -> str:
    """Build the user prompt for reference-free DermEval inference/training.

    Args:
        diagnostic_text: Candidate diagnostic narrative produced by a VLM/MLLM.

    Returns:
        A prompt that expects the model to look at the image and rate the
        candidate narrative across the six DermEval dimensions on a 0--5 scale.
    """
    diagnostic_text = str(diagnostic_text).strip()
    rubric_lines = "\n".join(
        f"- {name}: {RUBRIC_SHORT[key]}" for key, name in DERMEVAL_DIMENSIONS
    )
    schema = "\n".join(
        f"{name}: <score from 0 to 5> - <brief justification and one improvement suggestion>"
        for _, name in DERMEVAL_DIMENSIONS
    )
    return f"""You are DermEval, a dermatology-specific multimodal evaluator. Your task is to evaluate the candidate diagnostic narrative using ONLY the provided skin image and the candidate narrative. Do not use a gold reference answer and do not propose a replacement diagnosis.

Evaluate the narrative on a 0 to 5 scale for each dimension, where 0 is unusable and 5 is excellent.

Dimensions:
{rubric_lines}

Candidate Diagnostic Narrative:
{diagnostic_text}

Return exactly six titled sections in this machine-parsable format:
{schema}
""".strip()


def build_llava_conversations(
    diagnostic_text: str,
    evaluation_text: Optional[str] = None,
    include_image_token: bool = True,
) -> List[Dict[str, str]]:
    """Build one LLaVA conversation item for DermEval.

    The first message is a human/user message with the image token and the
    DermEval prompt. The optional second message is the assistant label used for
    supervised fine-tuning or teacher-forcing loss.
    """
    user_value = build_dermeval_prompt(diagnostic_text)
    if include_image_token and DEFAULT_IMAGE_TOKEN not in user_value:
        user_value = DEFAULT_IMAGE_TOKEN + "\n" + user_value
    conversations: List[Dict[str, str]] = [{"from": "human", "value": user_value}]
    if evaluation_text is not None:
        conversations.append({"from": "gpt", "value": str(evaluation_text).strip()})
    return conversations


def canonical_evaluation_text(
    scores: Mapping[str, float] | Sequence[float],
    rationales: Optional[Mapping[str, str]] = None,
    default_rationale: str = "The score reflects alignment between the image evidence and the candidate narrative.",
) -> str:
    """Create a parsable target evaluation text from numeric labels.

    Args:
        scores: Either a dict with canonical keys or a six-element sequence in
            DermEval order.
        rationales: Optional per-dimension text.
        default_rationale: Used when no rationale is supplied.
    """
    if isinstance(scores, Mapping):
        score_map = {key: float(scores[key]) for key in DERMEVAL_DIMENSION_KEYS}
    else:
        score_map = {key: float(value) for key, value in zip(DERMEVAL_DIMENSION_KEYS, scores)}
    rationales = rationales or {}
    lines: List[str] = []
    for key, name in DERMEVAL_DIMENSIONS:
        rationale = rationales.get(key) or rationales.get(name) or default_rationale
        lines.append(f"{name}: {score_map[key]:.1f} - {str(rationale).strip()}")
    overall = sum(score_map.values()) / len(score_map)
    lines.append(f"Overall Score: {overall:.1f}")
    return "\n".join(lines)


def summarize_scores(scores: Mapping[str, Optional[float]]) -> str:
    """Format parsed scores for logs or terminal output."""
    parts = []
    for key, name in DERMEVAL_DIMENSIONS:
        value = scores.get(key)
        parts.append(f"{name}: {'missing' if value is None else f'{value:.3g}'}")
    return "; ".join(parts)

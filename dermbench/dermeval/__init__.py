"""DermEval utilities for reference-free dermatology narrative evaluation.

This package supports reference-free dermatology diagnostic narrative evaluation:
image + candidate diagnostic narrative -> structured critique + six scores.
"""

from .constants import DERMEVAL_DIMENSIONS, DERMEVAL_DIMENSION_KEYS
from .prompts import build_dermeval_prompt, build_llava_conversations


def parse_scores(*args, **kwargs):
    from .score_parser import parse_scores as _parse_scores
    return _parse_scores(*args, **kwargs)


def reward_negative_mse(*args, **kwargs):
    from .score_parser import reward_negative_mse as _reward_negative_mse
    return _reward_negative_mse(*args, **kwargs)


__all__ = [
    "DERMEVAL_DIMENSIONS",
    "DERMEVAL_DIMENSION_KEYS",
    "build_dermeval_prompt",
    "build_llava_conversations",
    "parse_scores",
    "reward_negative_mse",
]

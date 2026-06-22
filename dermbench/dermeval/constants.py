"""Constants and compact rubrics for DermEval.

The six dimensions follow the DermBench/DermEval paper:
Accuracy, Safety, Medical Groundedness, Clinical Coverage,
Reasoning Coherence, and Description Precision.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

DERMEVAL_DIMENSIONS: List[Tuple[str, str]] = [
    ("accuracy", "Accuracy"),
    ("safety", "Safety"),
    ("medical_groundedness", "Medical Groundedness"),
    ("clinical_coverage", "Clinical Coverage"),
    ("reasoning_coherence", "Reasoning Coherence"),
    ("description_precision", "Description Precision"),
]

DERMEVAL_DIMENSION_KEYS: List[str] = [key for key, _ in DERMEVAL_DIMENSIONS]
DERMEVAL_DIMENSION_NAMES: List[str] = [name for _, name in DERMEVAL_DIMENSIONS]

# Aliases are intentionally broad because generated evaluators often use slightly
# different labels such as "Medical Factuality" or "Safety / Harmfulness".
DIMENSION_ALIASES: Dict[str, List[str]] = {
    "accuracy": ["Accuracy", "Acc", "Diagnostic Accuracy"],
    "safety": ["Safety", "Safety / Harmfulness", "Safety Harmfulness", "Harmfulness", "Patient Safety"],
    "medical_groundedness": [
        "Medical Groundedness",
        "Medical Factuality",
        "Medical Factuality Hallucination",
        "Groundedness",
        "Medical Correctness",
    ],
    "clinical_coverage": ["Clinical Coverage", "Coverage", "Completeness"],
    "reasoning_coherence": ["Reasoning Coherence", "Coherence", "Logical Coherence", "Reasoning"],
    "description_precision": ["Description Precision", "Descriptive Precision", "Description", "Precision"],
}

RUBRIC_SHORT: Dict[str, str] = {
    "accuracy": "Whether the diagnostic conclusion and main visual findings agree with expert judgment.",
    "safety": "Whether advice is conservative, avoids harm, and encourages appropriate clinical escalation.",
    "medical_groundedness": "Whether statements are supported by dermatology knowledge and avoid hallucinated claims.",
    "clinical_coverage": "Whether salient morphology, distribution, differentials, management, and follow-up are covered.",
    "reasoning_coherence": "Whether the reasoning follows logically from image observations to differential diagnosis and conclusion.",
    "description_precision": "Whether dermatologic morphology and distribution are described clearly and professionally.",
}

DEFAULT_SCORE_SCALE: float = 5.0

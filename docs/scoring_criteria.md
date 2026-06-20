# Scoring Criteria

DermBench uses six clinical dimensions, each scored from 0 to 5.

- Accuracy: agreement between the candidate diagnosis/key findings and the expert reference.
- Safety: potential for patient harm if the narrative is followed without expert supervision.
- Medical Groundedness: factual consistency with dermatology knowledge and clinical practice.
- Clinical Coverage: completeness of morphology, distribution, differential diagnosis, and follow-up/management considerations.
- Reasoning Coherence: internal logic from visual findings to differential diagnosis and conclusion.
- Description Precision: clarity and technical precision of dermatologic language.

Score interpretation:

- 5: high-quality, clinically aligned output with only minor issues.
- 4: broadly correct with minor omissions or secondary issues.
- 3: partially correct, with noticeable gaps or errors.
- 2: important information is missed, misstated, or confusing.
- 1: clinically unacceptable but still traceable to the case.
- 0: empty, off-topic, incoherent, or unusable for that dimension.


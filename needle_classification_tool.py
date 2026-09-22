"""Typed Friends and Family Test classification with Cactus Needle 3.

Usage:
    python needle_classification_tool.py "The receptionist was kind and helpful."
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Literal

# The local `needle/` directory contains model files and intentionally has no
# Python package. Remove the project directory while importing the installed
# cactus-needle package so it cannot shadow `import needle`.
_project_root = Path(__file__).resolve().parent
sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != _project_root]
import needle
from pydantic import BaseModel, Field


PrimaryTopic = Literal[
    "Appointment availability and lead times",
    "Telephone access",
    "Waiting time at the surgery",
    "Reception service",
    "Consultation communication and listening",
    "Clinical assessment and diagnosis",
    "Treatment effectiveness and safety",
    "Prescriptions and medication management",
    "Tests, investigations and results",
    "Referral, specialist and hospital coordination",
    "Follow-up and continuity of care",
    "Staff kindness and empathy",
    "Staff professionalism and knowledge",
    "Respect, dignity, privacy and inclusion",
    "Facilities, cleanliness and environment",
    "Physical accessibility",
    "Digital booking and online systems",
    "Mental health support",
    "Patient information and education",
    "Overall service and practice",
    "Other or unclassifiable",
]

SecondaryTopic = Literal[
    "Appointment availability and lead times",
    "Telephone access",
    "Waiting time at the surgery",
    "Reception service",
    "Consultation communication and listening",
    "Clinical assessment and diagnosis",
    "Treatment effectiveness and safety",
    "Prescriptions and medication management",
    "Tests, investigations and results",
    "Referral, specialist and hospital coordination",
    "Follow-up and continuity of care",
    "Staff kindness and empathy",
    "Staff professionalism and knowledge",
    "Respect, dignity, privacy and inclusion",
    "Facilities, cleanliness and environment",
    "Physical accessibility",
    "Digital booking and online systems",
    "Mental health support",
    "Patient information and education",
]

LOCAL_WEIGHTS = Path(__file__).parent / "needle" / "needle3.cact"


class ReviewClassification(BaseModel):
    """Structured output returned by Needle for one patient review."""

    sentiment: Literal["Positive", "Negative", "Neutral or mixed"] = Field(
        description="The overall emotional direction of the patient feedback."
    )
    primary_topic: PrimaryTopic = Field(
        description="The one main service topic receiving the greatest emphasis."
    )
    secondary_topics: list[SecondaryTopic] = Field(
        default_factory=list,
        description="Other topics independently and meaningfully mentioned in the review.",
    )
    actionability: Literal[
        "No actionable issue",
        "Possible improvement",
        "Clear improvement opportunity",
        "Specific practice-level issue",
    ] = Field(description="How directly the practice could act on the feedback.")
    urgency: Literal[
        "Routine",
        "Monitor",
        "Review soon",
        "Prompt review",
    ] = Field(description="How quickly a human should review the feedback.")
    safety_or_inclusion_concern: bool = Field(
        description=(
            "True only when the review explicitly raises patient safety, harm, "
            "discrimination, dignity, privacy, or inclusion concerns."
        )
    )


def classify_review(review: str) -> ReviewClassification | None:
    """Classify one review using Needle's structured extraction API."""
    if not review or not review.strip():
        return None
    weights = Path(os.environ.get("NEEDLE_WEIGHTS", LOCAL_WEIGHTS))
    extract_kwargs = {
        "text": review.strip(),
        "schema": ReviewClassification,
        "system": (
            "Classify a Friends and Family Test patient review. "
            "Use only information explicitly present in the review. "
            "Choose exactly one primary topic. Do not invent secondary topics."
        ),
    }
    if weights.exists():
        extract_kwargs["weights"] = str(weights)
    result = needle.extract(
        **extract_kwargs,
    )
    if result is not None:
        return result

    # Some Needle runtimes are stricter about extract() schemas. The direct
    # tool path gives the same typed Pydantic schema another supported route.
    agent_kwargs = {"tools": [ReviewClassification]}
    if weights.exists():
        agent_kwargs["weights"] = str(weights)
    agent = needle.Needle(**agent_kwargs)
    response = agent.complete(
        "Classify this Friends and Family Test review and populate the classification record.\n\n"
        f"Review: {review.strip()}",
        max_new_tokens=512,
    )
    calls = response.get("function_calls", [])
    if not calls:
        return None
    arguments = calls[0].get("arguments", {})
    return ReviewClassification.model_validate(arguments)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python needle_classification_tool.py \"review text\"")
    result = classify_review(sys.argv[1])
    if result is None:
        raise SystemExit("Needle returned no structured classification for this review.")
    print(json.dumps(result.model_dump() if result else None, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

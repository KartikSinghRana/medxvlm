#!/usr/bin/env python3
"""Hand-written assertion suite for `medxvlm.data.labeler`.

Exercises every rule the labeler implements -- plain mention detection,
bidirectional negation, pre/post hedging, scope-breaking conjunctions,
historical-qualifier demotion, and No Finding aggregation -- against
short, hand-written report sentences with known expected labels.

`scripts/prepare_data.py` imports and runs `main()` before labelling the
full corpus; it refuses to proceed if any case here fails.

Usage:
    python scripts/validate_labeler.py
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass

from medxvlm.data.labeler import NEGATIVE, NOT_MENTIONED, POSITIVE, UNCERTAIN, aggregate_report


@dataclass(frozen=True)
class Case:
    name: str
    text: str
    expected: dict[str, float]  # only the labels being asserted on


CASES: tuple[Case, ...] = (
    Case(
        "plain positive mention",
        "There is cardiomegaly.",
        {"Cardiomegaly": POSITIVE},
    ),
    Case(
        "pre-negation ('no X')",
        "There is no pneumothorax.",
        {"Pneumothorax": NEGATIVE},
    ),
    Case(
        "multi-word pre-negation cue ('no evidence of')",
        "No evidence of pneumothorax is seen today after examination and pneumonia is present.",
        {"Pneumothorax": NEGATIVE, "Pneumonia": POSITIVE},
    ),
    Case(
        "post-negation ('X is absent')",
        "Pleural effusion is absent.",
        {"Pleural Effusion": NEGATIVE},
    ),
    Case(
        "pre-modifier hedge ('possible X')",
        "Possible atelectasis is noted.",
        {"Atelectasis": UNCERTAIN},
    ),
    Case(
        "post-modifier hedge ('X cannot be excluded')",
        "Atelectasis cannot be excluded.",
        {"Atelectasis": UNCERTAIN},
    ),
    Case(
        "post-modifier hedge does not leak backward onto an earlier, unrelated mention",
        "There is cardiomegaly. Atelectasis cannot be excluded.",
        {"Cardiomegaly": POSITIVE, "Atelectasis": UNCERTAIN},
    ),
    Case(
        "scope-breaking conjunction blocks negation from crossing clauses",
        "No pneumothorax, but there is a pleural effusion.",
        {"Pneumothorax": NEGATIVE, "Pleural Effusion": POSITIVE},
    ),
    Case(
        "scope-breaking conjunction blocks hedging from crossing clauses",
        "Definite consolidation is present, although pneumothorax is unlikely.",
        {"Consolidation": POSITIVE},
    ),
    Case(
        "historical-qualifier demotion ('history of X')",
        "History of atelectasis.",
        {"Atelectasis": UNCERTAIN},
    ),
    Case(
        "historical qualifier does not demote an unrelated current finding",
        "History of atelectasis. Pneumonia is present.",
        {"Atelectasis": UNCERTAIN, "Pneumonia": POSITIVE},
    ),
    Case(
        "multiple positive mentions in one sentence",
        "Bibasilar atelectasis and small bilateral pleural effusions are noted.",
        {"Atelectasis": POSITIVE, "Pleural Effusion": POSITIVE},
    ),
    Case(
        "support device, plain positive",
        "There is a right IJ catheter and sternotomy wires are present.",
        {"Support Devices": POSITIVE},
    ),
    Case(
        "no mention at all",
        "The lungs are well expanded.",
        {"Pneumothorax": NOT_MENTIONED, "Fracture": NOT_MENTIONED},
    ),
    Case(
        "positive mention wins over an earlier negative one for the same label",
        "No definite cardiomegaly. Repeat view redemonstrates cardiomegaly.",
        {"Cardiomegaly": POSITIVE},
    ),
    Case(
        "No Finding true when every mention is negative or absent",
        "The heart is normal. There is no pneumothorax. No pleural effusion is seen. Lungs are clear.",
        {"No Finding": POSITIVE, "Pneumothorax": NEGATIVE, "Pleural Effusion": NEGATIVE},
    ),
    Case(
        "No Finding false when one finding is positive",
        "There is no pneumothorax. Mild cardiomegaly is present.",
        {"No Finding": NEGATIVE, "Cardiomegaly": POSITIVE},
    ),
    Case(
        "No Finding false when one finding is merely uncertain",
        "No pneumothorax. Possible atelectasis is noted.",
        {"No Finding": NEGATIVE, "Atelectasis": UNCERTAIN},
    ),
    # --- regression cases for the label-noise bugs found in Phase 5 ---
    Case(
        "leading negation governs the whole coordinated list, including the last item",
        "No focal consolidation, pleural effusion, or pneumothorax.",
        {"Consolidation": NEGATIVE, "Pleural Effusion": NEGATIVE, "Pneumothorax": NEGATIVE, "No Finding": POSITIVE},
    ),
    Case(
        "'without evidence of' governs a long coordinated list (real Indiana phrasing)",
        "The lungs are normally inflated without evidence of focal airspace disease, pleural effusion, or pneumothorax.",
        {"Pleural Effusion": NEGATIVE, "Pneumothorax": NEGATIVE},
    ),
    Case(
        "negation scope is reset by a new assertion inside the same clause",
        "There is no pneumothorax and there is a small left pleural effusion.",
        {"Pneumothorax": NEGATIVE, "Pleural Effusion": POSITIVE},
    ),
    Case(
        "'no' inside 'nodular' / 'normally' must not negate a later mention",
        "Nodular opacity in the right upper lobe.",
        {"Lung Opacity": POSITIVE},
    ),
    Case(
        "leading hedge governs a short coordinated list",
        "Possible atelectasis or effusion at the left base.",
        {"Atelectasis": UNCERTAIN, "Pleural Effusion": UNCERTAIN},
    ),
    Case(
        "trailing 'identified' after the list does not undo the leading negation",
        "No focal consolidation, pleural effusion, or pneumothorax identified.",
        {"Consolidation": NEGATIVE, "Pleural Effusion": NEGATIVE, "Pneumothorax": NEGATIVE},
    ),
)


def _values_equal(actual: float, expected: float) -> bool:
    if math.isnan(expected):
        return math.isnan(actual)
    return actual == expected


def main() -> int:
    failures: list[str] = []
    for case in CASES:
        result = aggregate_report(case.text)
        for label, expected in case.expected.items():
            actual = result[label]
            if not _values_equal(actual, expected):
                failures.append(
                    f"[{case.name}] label={label!r} text={case.text!r} "
                    f"expected={expected} actual={actual}"
                )

    print(f"Ran {len(CASES)} cases, {len(CASES) * 0 + sum(len(c.expected) for c in CASES)} assertions.")
    if failures:
        print(f"\nFAILED: {len(failures)} assertion(s):")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("OK: all labeler assertions passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

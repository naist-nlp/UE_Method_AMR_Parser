"""
Three curated few-shot examples for AMR parsing.

Fixed at 3 examples for all models and all prompt variants.
Selected to cover: simple action + coreference, named entity + time, negation.
All verified against AMR3.0 annotation guidelines.
"""

N_SHOTS = 3

EXAMPLES = [
    {
        "sentence": "The boy wants to go.",
        "amr": (
            "(w / want-01\n"
            "   :ARG0 (b / boy)\n"
            "   :ARG1 (g / go-02\n"
            "            :ARG0 b))"
        ),
    },
    {
        "sentence": "Obama visited China in 2013.",
        "amr": (
            "(v / visit-01\n"
            "   :ARG0 (p / person\n"
            "            :wiki \"Barack_Obama\"\n"
            "            :name (n / name :op1 \"Obama\"))\n"
            "   :ARG1 (c / country\n"
            "            :wiki \"China\"\n"
            "            :name (n2 / name :op1 \"China\"))\n"
            "   :time (d / date-entity :year 2013))"
        ),
    },
    {
        "sentence": "The girl did not sleep.",
        "amr": (
            "(s / sleep-01\n"
            "   :ARG0 (g / girl)\n"
            "   :polarity -)"
        ),
    },
]

# Kept for reference — not used in generation
_EXTRA_EXAMPLES = [
    {
        "sentence": "The doctor said the patient died because of the disease.",
        "amr": (
            "(s / say-01\n"
            "   :ARG0 (d / doctor)\n"
            "   :ARG1 (d2 / die-01\n"
            "             :ARG1 (p / patient)\n"
            "             :cause (d3 / disease)))"
        ),
    },
    {
        "sentence": "Three students passed the exam.",
        "amr": (
            "(p / pass-01\n"
            "   :ARG0 (s / student\n"
            "            :quant 3)\n"
            "   :ARG1 (e / exam))"
        ),
    },
]


def build_few_shot_block() -> str:
    """Return the 3 fixed examples as a plain-text block for the prompt."""
    parts = []
    for i, ex in enumerate(EXAMPLES, start=1):
        parts.append(f"Example {i}:\nSentence: {ex['sentence']}\nAMR:\n{ex['amr']}")
    return "\n\n".join(parts)

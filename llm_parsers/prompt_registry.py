"""
Prompt variant registry for AMR parsing experiments.

Prompts are defined in prompts.txt and mirrored here as PromptVariant entries.
Each prompt is a self-contained template with {sentence} as the only placeholder.
Few-shot examples are embedded directly in each template.

Temperature is fixed at 0.7 for all runs.

Variant IDs are included in the output directory name and results.json key:
  data/k_top_AMR_parsers/{model_label}_{prompt_id}/

Current variants (from prompts.txt)
------------------------------------
p1  Rules + 4-step internal procedure + 3 examples. Direct AMR output.
p2  Step-by-step construction strategy + 3 examples. Direct AMR output.
p3  Chain-of-thought: shows Reasoning + AMR in examples; model outputs both.
p4  Analyse-then-output: think internally, output only AMR.
"""

from typing import TypedDict


class PromptVariant(TypedDict):
    id: str
    description: str
    template: str        # full prompt with {sentence} as only placeholder
    forced_prefix: str   # prepended to raw model output before AMR extraction
    extract_after: str   # if non-empty, extract AMR from after this marker in output


_VARIANTS: list[PromptVariant] = [

    # ── p1: Rules + 4-step internal procedure, direct output ─────────────────
    {
        "id": "p1",
        "description": "Rules + 4-step internal procedure, direct AMR output",
        "template": (
            "You are an expert Abstract Meaning Representation (AMR) parser.\n\n"
            "Your task is to convert a sentence into a valid AMR graph.\n\n"
            "Important Rules:\n"
            "- Output MUST contain ONLY the AMR graph.\n"
            "- Do NOT explain the reasoning.\n"
            "- Do NOT generate natural language text.\n"
            "- Use valid PENMAN notation only.\n"
            "- Use PropBank frames when appropriate.\n"
            "- Reuse variables for shared entities and coreference.\n"
            "- Use :polarity - for negation.\n"
            "- Use :wiki and :name for named entities.\n\n"
            "Follow this parsing structure internally:\n"
            "1. Identify the main predicate.\n"
            "2. Identify semantic roles and relations.\n"
            "3. Detect entities, dates, quantities, and modifiers.\n"
            "4. Construct the final AMR graph.\n\n"
            "Examples:\n\n"
            "Sentence: The boy wants to go.\n"
            "AMR:\n"
            "(w / want-01\n"
            "   :ARG0 (b / boy)\n"
            "   :ARG1 (g / go-02\n"
            "            :ARG0 b))\n\n"
            "Sentence: Obama visited China in 2013.\n"
            "AMR:\n"
            "(v / visit-01\n"
            "   :ARG0 (p / person\n"
            "            :wiki \"Barack_Obama\"\n"
            "            :name (n / name :op1 \"Obama\"))\n"
            "   :ARG1 (c / country\n"
            "            :wiki \"China\"\n"
            "            :name (n2 / name :op1 \"China\"))\n"
            "   :time (d / date-entity :year 2013))\n\n"
            "Sentence: The girl did not sleep.\n"
            "AMR:\n"
            "(s / sleep-01\n"
            "   :ARG0 (g / girl)\n"
            "   :polarity -)\n\n"
            "Sentence: {sentence}\n"
            "AMR:\n"
        ),
        "forced_prefix": "",
        "extract_after": "",
    },

    # ── p2: Step-by-step construction strategy, direct output ────────────────
    {
        "id": "p2",
        "description": "5-step construction strategy, direct AMR output",
        "template": (
            "You are a semantic parsing system specialized in Abstract Meaning Representation (AMR).\n\n"
            "Goal:\n"
            "Generate a complete semantic graph for the input sentence.\n\n"
            "Output Constraints:\n"
            "- The output MUST be AMR only.\n"
            "- Return ONLY a valid PENMAN-style AMR graph.\n"
            "- Do NOT provide explanations or reasoning text.\n"
            "- Do NOT generate markdown or extra formatting.\n"
            "- Ensure parentheses and variables are valid.\n\n"
            "AMR Construction Strategy:\n"
            "Step 1: Detect the main event or predicate.\n"
            "Step 2: Identify participants and semantic roles.\n"
            "Step 3: Detect modifiers, polarity, quantities, dates, and named entities.\n"
            "Step 4: Resolve coreference and variable reuse.\n"
            "Step 5: Construct the final AMR graph.\n\n"
            "Few-shot Examples:\n\n"
            "Sentence: The boy wants to go.\n"
            "AMR:\n"
            "(w / want-01\n"
            "   :ARG0 (b / boy)\n"
            "   :ARG1 (g / go-02\n"
            "            :ARG0 b))\n\n"
            "Sentence: Obama visited China in 2013.\n"
            "AMR:\n"
            "(v / visit-01\n"
            "   :ARG0 (p / person\n"
            "            :wiki \"Barack_Obama\"\n"
            "            :name (n / name :op1 \"Obama\"))\n"
            "   :ARG1 (c / country\n"
            "            :wiki \"China\"\n"
            "            :name (n2 / name :op1 \"China\"))\n"
            "   :time (d / date-entity :year 2013))\n\n"
            "Sentence: The girl did not sleep.\n"
            "AMR:\n"
            "(s / sleep-01\n"
            "   :ARG0 (g / girl)\n"
            "   :polarity -)\n\n"
            "Now generate the AMR graph.\n\n"
            "Sentence: {sentence}\n"
            "AMR:\n"
        ),
        "forced_prefix": "",
        "extract_after": "",
    },

    # ── p3: Chain-of-thought — shows Reasoning in examples, model outputs both ─
    {
        "id": "p3",
        "description": "Chain-of-thought: Reasoning shown in examples, model continues Reasoning then AMR",
        "template": (
            "You are an expert Abstract Meaning Representation (AMR) parser.\n\n"
            "Convert the sentence into a valid AMR graph.\n\n"
            "Instructions:\n"
            "- Think step by step internally before generating the AMR.\n"
            "- Identify the main predicate first.\n"
            "- Then identify semantic roles and relations.\n"
            "- Detect entities, dates, negation, and quantities.\n"
            "- Reuse variables for shared entities.\n"
            "- Output ONLY the final AMR graph.\n"
            "- Do NOT output explanations or reasoning text.\n\n"
            "Examples:\n\n"
            "Sentence: The boy wants to go.\n"
            "Reasoning:\n"
            "Main predicate is want-01.\n"
            "The boy is ARG0 of want-01.\n"
            "go-02 is ARG1 of want-01.\n"
            "The boy is also ARG0 of go-02.\n\n"
            "AMR:\n"
            "(w / want-01\n"
            "   :ARG0 (b / boy)\n"
            "   :ARG1 (g / go-02\n"
            "            :ARG0 b))\n\n"
            "Sentence: Obama visited China in 2013.\n"
            "Reasoning:\n"
            "Main predicate is visit-01.\n"
            "Obama is ARG0.\n"
            "China is ARG1.\n"
            "2013 is a time expression.\n\n"
            "AMR:\n"
            "(v / visit-01\n"
            "   :ARG0 (p / person\n"
            "            :wiki \"Barack_Obama\"\n"
            "            :name (n / name :op1 \"Obama\"))\n"
            "   :ARG1 (c / country\n"
            "            :wiki \"China\"\n"
            "            :name (n2 / name :op1 \"China\"))\n"
            "   :time (d / date-entity :year 2013))\n\n"
            "Sentence: The girl did not sleep.\n"
            "Reasoning:\n"
            "Main predicate is sleep-01.\n"
            "The girl is ARG0.\n"
            "Sentence contains negation.\n\n"
            "AMR:\n"
            "(s / sleep-01\n"
            "   :ARG0 (g / girl)\n"
            "   :polarity -)\n\n"
            "Sentence: {sentence}\n"
            "Reasoning:\n"
            "AMR:\n"
        ),
        "forced_prefix": "",
        # model may output "Reasoning: ...\nAMR:\n(..." — extract from after last "AMR:"
        "extract_after": "AMR:",
    },

    # ── p4: Analyse internally, output only AMR ───────────────────────────────
    {
        "id": "p4",
        "description": "Analyse-then-output: think internally, output only the AMR graph",
        "template": (
            "You are a semantic parser specialized in Abstract Meaning Representation (AMR).\n\n"
            "Your task is to generate a valid AMR graph for the given sentence.\n\n"
            "Before generating the graph:\n"
            "- Analyze the sentence carefully.\n"
            "- Think step by step internally.\n"
            "- Identify predicates, arguments, modifiers, entities, and relations.\n"
            "- Resolve reentrancies and shared arguments.\n\n"
            "Output Rules:\n"
            "- Output MUST contain ONLY the AMR graph.\n"
            "- Do NOT generate reasoning steps.\n"
            "- Do NOT explain the answer.\n"
            "- Use valid PENMAN notation only.\n\n"
            "Few-shot Examples:\n\n"
            "Sentence: The boy wants to go.\n"
            "AMR:\n"
            "(w / want-01\n"
            "   :ARG0 (b / boy)\n"
            "   :ARG1 (g / go-02\n"
            "            :ARG0 b))\n\n"
            "Sentence: Obama visited China in 2013.\n"
            "AMR:\n"
            "(v / visit-01\n"
            "   :ARG0 (p / person\n"
            "            :wiki \"Barack_Obama\"\n"
            "            :name (n / name :op1 \"Obama\"))\n"
            "   :ARG1 (c / country\n"
            "            :wiki \"China\"\n"
            "            :name (n2 / name :op1 \"China\"))\n"
            "   :time (d / date-entity :year 2013))\n\n"
            "Sentence: The girl did not sleep.\n"
            "AMR:\n"
            "(s / sleep-01\n"
            "   :ARG0 (g / girl)\n"
            "   :polarity -)\n\n"
            "Sentence: {sentence}\n"
            "AMR:\n"
        ),
        "forced_prefix": "",
        "extract_after": "",
    },
]

PROMPT_REGISTRY: dict[str, PromptVariant] = {v["id"]: v for v in _VARIANTS}
PROMPT_IDS = list(PROMPT_REGISTRY.keys())
DEFAULT_PROMPT_ID = "p1"


def get_prompt(prompt_id: str) -> PromptVariant:
    if prompt_id not in PROMPT_REGISTRY:
        raise ValueError(f"Unknown prompt id '{prompt_id}'. Available: {PROMPT_IDS}")
    return PROMPT_REGISTRY[prompt_id]


def describe_all() -> None:
    print(f"{'ID':<6} Description")
    print("─" * 70)
    for v in _VARIANTS:
        print(f"{v['id']:<6} {v['description']}")

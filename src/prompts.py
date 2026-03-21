DEEPSEEK_PARAPHRASE_PROMPT = """You are given a one-sentence factual news summary.

Rewrite it into FOUR different one-sentence paraphrases.
Each paraphrase MUST:
- preserve exactly the same facts, entities, and meaning,
- NOT add or remove any information,
- use different wording and sentence structure,
- remain a single sentence,
- be distinct from the other paraphrases,
- not be identical to the original summary.

Do NOT explain anything.
Do NOT number the sentences.
Return the four paraphrases as separate lines.

Original summary:
"{SUMMARY}"
"""

# Fixed prompt templates for the Contamination Detection Pipeline (API-only).
# Keep these templates stable across runs and across evaluated models for comparability.

# -----------------------------
# 1) DCQ (Semantic) detector
# -----------------------------
# Expected model output: a single letter "A", "B", "C", "D", or "E"
# (no punctuation, no explanation). E means "None of the provided options".
# Variables:
#   {DOCUMENT}  - full document text (normalized or raw, but consistent across runs)
#   {A}, {B}, {C}, {D} - candidate one-sentence summaries (canonical + sampled paraphrases, shuffled)
#   {E} - fixed "None of the provided options"
DCQ_PROMPT_TEMPLATE = """Choose the best ONE-SENTENCE summary of the article.
Reply with A, B, C, D, or E only. Do not add any other text.

Article:
{DOCUMENT}

Options:
A. {A}
B. {B}
C. {C}
D. {D}
E. {E}
"""


# -----------------------------
# 2) Memorization probing detector
# -----------------------------
# Goal: elicit continuation that matches the canonical reference summary.
# Expected output: free-form continuation text (typically short).
# Variables:
#   {PREFIX} - reference prefix (frozen)
MEM_PROMPT_TEMPLATE = """Instruction: You are provided with the first piece of a one-sentence summary from the {SPLIT_NAME} split of the XSum dataset.
Finish the second piece of the summary as exactly appeared in the dataset.
Only rely on the original form of the summary in the dataset to finish the second piece.

First Piece: {PREFIX}
Second Piece:
"""


# -----------------------------
# 3) Stability / Probability detector (N-sampling)
# -----------------------------
# Goal: generate multiple stochastic one-sentence summaries for the same document.
# Expected output: exactly one sentence.
# Variables:
#   {DOCUMENT} - full document text
STABILITY_PROMPT_TEMPLATE = """Write a ONE-SENTENCE summary of the following news article.
Keep it factual and concise. Output exactly one sentence.

Article:
{DOCUMENT}
"""

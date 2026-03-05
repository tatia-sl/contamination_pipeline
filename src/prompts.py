DEEPSEEK_PARAPHRASE_PROMPT = """You are given a one-sentence factual news summary.

Rewrite it into THREE different one-sentence paraphrases.
Each paraphrase MUST:
- preserve exactly the same facts, entities, and meaning,
- NOT add or remove any information,
- use different wording and sentence structure,
- remain a single sentence.

Do NOT explain anything.
Do NOT number the sentences.
Return the three paraphrases as separate lines.

Original summary:
"{SUMMARY}"
"""

# Fixed prompt templates for the Contamination Detection Pipeline (API-only).
# Keep these templates stable across runs and across evaluated models for comparability.

# -----------------------------
# 1) DCQ (Semantic) detector
# -----------------------------
# Expected model output: a single letter "A", "B", "C", or "D" (no punctuation, no explanation).
# Variables:
#   {DOCUMENT}  - full document text (normalized or raw, but consistent across runs)
#   {A}, {B}, {C}, {D} - candidate one-sentence summaries (canonical + 3 paraphrases, shuffled)
DCQ_PROMPT_TEMPLATE = """Choose the best ONE-SENTENCE summary of the article.
Reply with A, B, C, or D only. Do not add any other text.

Article:
{DOCUMENT}

Options:
A. {A}
B. {B}
C. {C}
D. {D}
"""


# -----------------------------
# 2) Memorization probing detector
# -----------------------------
# Goal: elicit continuation that matches the canonical reference summary.
# Expected output: free-form continuation text (typically short).
# Variables:
#   {PREFIX} - reference prefix (frozen)
MEM_PROMPT_TEMPLATE = """Complete the following sentence exactly as it appears in the original reference summary.
Continue from where it stops. Do not add any extra sentences.

Prefix:
{PREFIX}
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

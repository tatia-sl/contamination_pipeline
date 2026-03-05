import os
from typing import Optional
from openai import OpenAI


class OpenAIClient:
    def __init__(self, model: str, api_key: Optional[str] = None):
        """
        Minimal OpenAI client used by run_dcq_detector.

        Args:
            model: model name, e.g., "gpt-4o-mini"
            api_key: optional override; falls back to OPENAI_API_KEY env var
        """
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("Missing OPENAI_API_KEY env var")

        self.client = OpenAI(api_key=key)
        self.model = model

    def generate_text(self, prompt: str, temperature: float, top_p: float, max_tokens: int) -> str:
        """
        Generate text via Chat Completions for compatibility with rest of pipeline.
        """
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        return (resp.choices[0].message.content or "").strip()

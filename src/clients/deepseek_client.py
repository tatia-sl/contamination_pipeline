import os
from openai import OpenAI

class DeepSeekClient:
    """
    Minimal wrapper around DeepSeek OpenAI-compatible Chat Completions API.
    Docs confirm OpenAI-compatible format and base_url usage. :contentReference[oaicite:6]{index=6}
    """
    def __init__(self, model: str = "deepseek-chat", base_url: str = "https://api.deepseek.com/v1"):
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("Missing DEEPSEEK_API_KEY env var")

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def generate_text(self, prompt: str, temperature: float = 0.4, top_p: float = 0.9) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_p,
        )
        return resp.choices[0].message.content

import os
from google import genai

class GeminiClient:
    def __init__(self, model: str):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GEMINI_API_KEY env var")
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def generate_text(self, prompt: str, temperature: float, top_p: float, max_tokens: int) -> str:
        resp = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "temperature": temperature,
                "top_p": top_p,
                "max_output_tokens": max_tokens,
            }
        )
        return (resp.text or "").strip()

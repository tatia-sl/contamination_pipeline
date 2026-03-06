import os
from google import genai


class GeminiClient:
    def __init__(self, model: str):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GEMINI_API_KEY env var")
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.last_response_meta = {}

    def generate_text(self, prompt: str, temperature: float, top_p: float, max_tokens: int) -> str:
        gen_config = {
            "temperature": temperature,
            "top_p": top_p,
            "max_output_tokens": max_tokens,
        }
        # For Gemini 2.5 family, disable thinking budget to avoid empty outputs in short-answer tasks.
        if "gemini-2.5" in self.model:
            gen_config["thinking_config"] = {"thinking_budget": 0}

        resp = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=gen_config
        )
        candidates = getattr(resp, "candidates", None) or []
        prompt_feedback = getattr(resp, "prompt_feedback", None)
        self.last_response_meta = {
            "provider": "gemini",
            "model": self.model,
            "prompt_feedback_block_reason": str(getattr(prompt_feedback, "block_reason", None)),
            "prompt_feedback": str(prompt_feedback),
            "candidate_finish_reasons": [str(getattr(c, "finish_reason", None)) for c in candidates],
            "candidate_safety_ratings": [str(getattr(c, "safety_ratings", None)) for c in candidates],
        }

        # Some Gemini models may return empty resp.text while keeping output in candidate parts.
        text = (getattr(resp, "text", None) or "").strip()
        self.last_response_meta["resp_text_present"] = bool(text)
        if text:
            return text

        parts_text = []
        for cand in candidates:
            content = getattr(cand, "content", None)
            parts = getattr(content, "parts", None) or []
            for part in parts:
                t = getattr(part, "text", None)
                if isinstance(t, str) and t.strip():
                    parts_text.append(t.strip())
        self.last_response_meta["parts_text_count"] = len(parts_text)
        if parts_text:
            return " ".join(parts_text).strip()

        return ""

import os
from typing import Optional, Dict
from openai import OpenAI


def _sanitize_openai_api_key(raw_key: str) -> str:
    """
    Normalize common copy-paste artifacts in API keys.
    """
    key = (raw_key or "").strip().strip('"').strip("'")
    # Replace common Unicode dashes with ASCII '-'
    key = key.replace("—", "-").replace("–", "-").replace("−", "-")
    # Remove accidental spaces/newlines inside the token
    key = "".join(key.split())
    return key


class OpenAIClient:
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ):
        """
        Minimal OpenAI client used by run_dcq_detector.

        Args:
            model: model name, e.g., "gpt-4o-mini"
            api_key: optional override; falls back to OPENAI_API_KEY env var
        """
        key = _sanitize_openai_api_key(api_key or os.environ.get("OPENAI_API_KEY", ""))
        if not key:
            raise RuntimeError("Missing OPENAI_API_KEY env var")
        try:
            key.encode("ascii")
        except UnicodeEncodeError as exc:
            raise RuntimeError(
                "OPENAI_API_KEY contains non-ASCII characters. Re-copy the key and ensure only standard ASCII symbols."
            ) from exc

        kwargs = {"api_key": key}
        if base_url:
            kwargs["base_url"] = base_url
        if extra_headers:
            kwargs["default_headers"] = extra_headers

        self.client = OpenAI(**kwargs)
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

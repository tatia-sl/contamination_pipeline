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
        api_mode: str = "chat_completions",
    ):
        """
        Minimal OpenAI-compatible client used by detector scripts
        (DCQ, memorization, stability), including OpenRouter base_url mode.

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
        self.api_mode = (api_mode or "chat_completions").strip().lower()
        self.last_response_meta: Optional[Dict[str, str]] = None

    @staticmethod
    def _extract_responses_text(resp) -> str:
        # Newer SDKs expose `output_text`; keep fallback parsing for compatibility.
        text = getattr(resp, "output_text", "") or ""
        if text:
            return text.strip()
        chunks = []
        for item in getattr(resp, "output", []) or []:
            for part in getattr(item, "content", []) or []:
                ptype = getattr(part, "type", None)
                if ptype in {"output_text", "text"}:
                    val = getattr(part, "text", "") or ""
                    if val:
                        chunks.append(val)
                elif ptype == "refusal":
                    val = getattr(part, "refusal", "") or ""
                    if val:
                        chunks.append(val)

        if chunks:
            return "\n".join(chunks).strip()

        # Last-resort extraction for SDK/shape differences.
        dumped = None
        if hasattr(resp, "model_dump"):
            dumped = resp.model_dump()
        elif hasattr(resp, "to_dict"):
            dumped = resp.to_dict()
        if not isinstance(dumped, dict):
            return ""

        found = []

        def walk(node):
            if isinstance(node, dict):
                for k, v in node.items():
                    if k in {"text", "refusal"} and isinstance(v, str) and v.strip():
                        found.append(v.strip())
                    else:
                        walk(v)
            elif isinstance(node, list):
                for x in node:
                    walk(x)

        walk(dumped)
        return "\n".join(found).strip()

    def generate_text(self, prompt: str, temperature: float, top_p: float, max_tokens: int) -> str:
        """
        Generate text via configured OpenAI endpoint.
        """
        if self.api_mode == "completions":
            resp = self.client.completions.create(
                model=self.model,
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            text = (resp.choices[0].text or "") if getattr(resp, "choices", None) else ""
            return text.strip()

        if self.api_mode == "responses":
            lower_prompt = (prompt or "").lower()
            dcq_letter_mode = (
                "reply with a, b, c, d, or e" in lower_prompt
                or "exactly one letter only: a, b, c, d, or e" in lower_prompt
            )
            response_max_tokens = max(int(max_tokens), 512)
            resp = self.client.responses.create(
                model=self.model,
                input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
                instructions=(
                    "Return exactly one uppercase letter: A, B, C, D, or E."
                    if dcq_letter_mode
                    else None
                ),
                max_output_tokens=response_max_tokens,
                reasoning={"effort": "medium"},
                text={"verbosity": "low", "format": {"type": "text"}},
            )
            text = self._extract_responses_text(resp)
            if text:
                return text

            self.last_response_meta = {
                "status": str(getattr(resp, "status", "")),
                "incomplete_details": str(getattr(resp, "incomplete_details", "")),
            }

            # Fallback: retry with minimal payload for maximum compatibility.
            resp2 = self.client.responses.create(
                model=self.model,
                input=prompt,
                max_output_tokens=max(256, max_tokens),
            )
            text2 = self._extract_responses_text(resp2)
            if text2:
                return text2

            self.last_response_meta = {
                "status": str(getattr(resp2, "status", "")),
                "incomplete_details": str(getattr(resp2, "incomplete_details", "")),
            }
            return ""

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()

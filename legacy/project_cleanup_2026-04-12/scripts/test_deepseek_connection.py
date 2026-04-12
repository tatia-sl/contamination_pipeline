import os
from openai import OpenAI

def main():
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("Missing DEEPSEEK_API_KEY env var")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",  # OpenAI-compatible base_url :contentReference[oaicite:4]{index=4}
    )

    resp = client.chat.completions.create(
        model="deepseek-chat",  # DeepSeek-V3 is exposed via model='deepseek-chat' :contentReference[oaicite:5]{index=5}
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        temperature=0.0,
    )

    print(resp.choices[0].message.content)

if __name__ == "__main__":
    main()

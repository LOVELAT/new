from __future__ import annotations

from typing import Optional

import openai


def get_llm_response(
    client: openai.OpenAI,
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    model: str = "qwen3-vl-plus",
    temperature: float = 0.4,
    top_p: float = 0.95,
    max_tokens: int = 900,
) -> str:
    """Get a text response from a chat completion model."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    content: Optional[str] = response.choices[0].message.content
    if content is None:
        return ""
    return content

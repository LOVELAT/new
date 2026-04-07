from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable


def load_prompts_from_file(file_path: str) -> Dict[str, str]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Prompts file not found: {path}")

    prompts: Dict[str, str] = {}
    current_prompt: str | None = None
    current_content: list[str] = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("[") and line.endswith("]"):
            if current_prompt:
                prompts[current_prompt] = "\n".join(current_content).strip()
            current_prompt = line[1:-1]
            current_content = []
        elif line:
            current_content.append(line)

    if current_prompt:
        prompts[current_prompt] = "\n".join(current_content).strip()

    return prompts


def load_tool_prompts(tools: Iterable[str], tools_json_path: str) -> str:
    path = Path(tools_json_path)
    if not path.exists():
        raise FileNotFoundError(f"Tools JSON file not found: {path}")

    tools_data = json.loads(path.read_text(encoding="utf-8"))
    tool_prompts: list[str] = []

    for tool in tools:
        if tool not in tools_data:
            continue
        tool_info = tools_data[tool]
        tool_prompts.append(
            "\n".join(
                [
                    f"Tool: {tool}",
                    f"Description: {tool_info['description']}",
                    f"Usage: {tool_info['prompt']}",
                    f"Input type: {tool_info['input_type']}",
                    f"Return type: {tool_info['return_type']}",
                ]
            )
        )

    return "\n\n".join(tool_prompts)


def load_system_prompt(
    system_prompts_file: str,
    system_prompt_type: str,
    tools: Iterable[str],
    tools_json_path: str,
) -> str:
    prompts = load_prompts_from_file(system_prompts_file)
    system_prompt = prompts.get(system_prompt_type, prompts.get("GENERAL_ASSISTANT", ""))
    tool_prompts = load_tool_prompts(tools, tools_json_path)

    if not tool_prompts:
        return system_prompt.strip()
    return f"{system_prompt}\n\nTools:\n{tool_prompts}".strip()

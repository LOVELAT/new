"""Tool registry for the baggage agent.

Add real tool constructors to `TOOL_BUILDERS` as the project grows.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional

from langchain_core.tools import BaseTool

TOOL_BUILDERS: Dict[str, Callable[..., BaseTool]] = {}


def build_tools(
    tools_to_use: Optional[Iterable[str]] = None,
    **shared_kwargs,
) -> Dict[str, BaseTool]:
    """Instantiate only the requested tools.

    The registry is intentionally empty for now, matching the current
    requirement that the toolbar should stay blank.
    """

    if tools_to_use is None:
        selected_names = list(TOOL_BUILDERS.keys())
    else:
        selected_names = list(tools_to_use)

    tools: Dict[str, BaseTool] = {}
    for tool_name in selected_names:
        builder = TOOL_BUILDERS.get(tool_name)
        if builder is None:
            continue
        tools[tool_name] = builder(**shared_kwargs)

    return tools


__all__ = ["TOOL_BUILDERS", "build_tools"]

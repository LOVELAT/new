from __future__ import annotations

import json
import operator
from datetime import datetime
from pathlib import Path
from typing import Any, Annotated, Dict, List, Optional, TypedDict

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph

if load_dotenv is not None:
    _ = load_dotenv()


class ToolCallLog(TypedDict):
    """Serialized tool call log entry."""

    timestamp: str
    tool_call_id: str
    name: str
    args: Any
    content: str


class AgentState(TypedDict):
    """Conversation state tracked by the workflow."""

    messages: Annotated[List[AnyMessage], operator.add]


class Agent:
    """Small LangGraph agent modeled after the MedRAX workflow."""

    def __init__(
        self,
        model: BaseLanguageModel,
        tools: List[BaseTool],
        checkpointer: Any = None,
        system_prompt: str = "",
        log_tools: bool = True,
        log_dir: Optional[str] = "logs",
    ) -> None:
        self.system_prompt = system_prompt
        self.log_tools = log_tools
        self.tools = {tool.name: tool for tool in tools}
        self.model = model.bind_tools(tools)

        if self.log_tools:
            self.log_path = Path(log_dir or "logs")
            self.log_path.mkdir(parents=True, exist_ok=True)

        workflow = StateGraph(AgentState)
        workflow.add_node("process", self.process_request)
        workflow.add_node("execute", self.execute_tools)
        workflow.add_conditional_edges(
            "process",
            self.has_tool_calls,
            {True: "execute", False: END},
        )
        workflow.add_edge("execute", "process")
        workflow.set_entry_point("process")
        self.workflow = workflow.compile(checkpointer=checkpointer)

    def process_request(self, state: AgentState) -> Dict[str, List[AnyMessage]]:
        messages = state["messages"]
        if self.system_prompt:
            messages = [SystemMessage(content=self.system_prompt)] + messages

        response = self.model.invoke(messages)
        return {"messages": [response]}

    def has_tool_calls(self, state: AgentState) -> bool:
        response = state["messages"][-1]
        return len(getattr(response, "tool_calls", []) or []) > 0

    def execute_tools(self, state: AgentState) -> Dict[str, List[ToolMessage]]:
        tool_calls = getattr(state["messages"][-1], "tool_calls", []) or []
        results: List[ToolMessage] = []

        for call in tool_calls:
            tool_name = call["name"]
            if tool_name not in self.tools:
                result: Any = {
                    "error": f"Unknown tool '{tool_name}'. Please choose another tool.",
                }
            else:
                result = self.tools[tool_name].invoke(call.get("args", {}))

            results.append(
                ToolMessage(
                    tool_call_id=call["id"],
                    name=tool_name,
                    args=call.get("args", {}),
                    content=self._serialize_tool_result(result),
                )
            )

        self._save_tool_calls(results)
        return {"messages": results}

    @staticmethod
    def _serialize_tool_result(result: Any) -> str:
        if isinstance(result, str):
            return result

        try:
            return json.dumps(result, ensure_ascii=False, indent=2)
        except TypeError:
            return str(result)

    def _save_tool_calls(self, tool_calls: List[ToolMessage]) -> None:
        if not self.log_tools:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.log_path / f"tool_calls_{timestamp}.json"

        logs: List[ToolCallLog] = []
        for call in tool_calls:
            logs.append(
                {
                    "tool_call_id": call.tool_call_id,
                    "name": call.name,
                    "args": call.args,
                    "content": str(call.content),
                    "timestamp": datetime.now().isoformat(),
                }
            )

        with filename.open("w", encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)

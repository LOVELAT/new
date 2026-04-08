from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

from dotenv import load_dotenv
from transformers import logging as transformers_logging
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from .agent import Agent
from .tools import build_tools
from .utils import load_prompts_from_file

warnings.filterwarnings("ignore")
transformers_logging.set_verbosity_error()
_ = load_dotenv()


def initialize_agent(
    prompt_file: str | os.PathLike[str] = "baggage/docs/system_prompts.txt",
    tools_to_use: Optional[Iterable[str]] = None,
    model_dir: str = "/model-weights",
    temp_dir: str = "temp",
    device: str = "cuda",
    model: str = "gpt-4o-mini",
    temperature: float = 0.3,
    top_p: float = 0.95,
    openai_kwargs: Optional[Dict[str, object]] = None,
) -> Tuple[Agent, Dict[str, BaseTool]]:
    """Initialize the baggage agent using the MedRAX-style layout."""

    prompts = load_prompts_from_file(str(prompt_file))
    prompt = prompts["BAGGAGE_ASSISTANT"]
    tools_dict = build_tools(
        tools_to_use=tools_to_use,
        model_dir=model_dir,
        temp_dir=temp_dir,
        device=device,
    )

    checkpointer = MemorySaver()
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        top_p=top_p,
        **(openai_kwargs or {}),
    )

    agent = Agent(
        llm,
        tools=list(tools_dict.values()),
        log_tools=True,
        log_dir="logs",
        system_prompt=prompt,
        checkpointer=checkpointer,
    )
    return agent, tools_dict


if __name__ == "__main__":
    from .interface import create_demo

    selected_tools: list[str] = ["cspcl_detector"]

    openai_kwargs: Dict[str, object] = {}
    if api_key := os.getenv("OPENAI_API_KEY"):
        openai_kwargs["api_key"] = api_key
    if base_url := os.getenv("OPENAI_BASE_URL"):
        openai_kwargs["base_url"] = base_url

    prompt_path = Path(__file__).resolve().parent / "docs" / "system_prompts.txt"
    agent, tools_dict = initialize_agent(
        prompt_file=str(prompt_path),
        tools_to_use=selected_tools,
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.3,
        top_p=0.95,
        openai_kwargs=openai_kwargs,
    )
    demo = create_demo(agent, tools_dict)
    demo.launch(server_name="0.0.0.0", server_port=8585, share=False)

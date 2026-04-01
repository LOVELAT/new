#!/usr/bin/env python3
"""
Baggage X-ray benchmark generator based on testray metadata.

This script mirrors the MedRAX benchmark workflow, but is tailored for
aviation baggage threat detection and uses only `testray_metadata.json`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import openai
from tqdm import tqdm
from llm import get_llm_response
from utils import load_testray_dataset


DEFAULT_DATASET_PATH = r"F:\Project\new\testray\testray_metadata.json"
DEFAULT_OUTPUT_DIR = r"F:\Project\new\baggage_benchmark\questions"

SYSTEM_PROMPT = (
    "You are an expert benchmark creation assistant for airport baggage X-ray analysis. "
    "Create challenging, verifiable multiple-choice questions based strictly on provided metadata."
)

CATEGORIES_META = {
    "detection": "Analyze the X-ray image, first determining whether prohibited items are present, and if so, further classifying them into security categories.",
    "Counting": "Count target objects based on annotations.",
    "localization": "Determine object location based on bounding-box coordinates.",
    "relationship": "Reason about overlap, occlusion relation between objects.",
    "risk_assessment": "Assess operational risk level from object combinations.",
    "decision": "Choose the most appropriate screening follow-up action.",
    "reasoning": "Explain the rationale using only the provided case evidence.",
}

CATEGORY_COMBINATIONS = [
    ["detection", "localization", "reasoning"],
    ["detection", "Counting", "reasoning"],
    ["localization", "relationship", "reasoning"],
    ["detection", "risk_assessment", "decision", "reasoning"],
]

DEFAULT_SECTIONS = [
    "history",
    "image_finding",
    "diagnosis",
    "labels",
    "object_count",
    "features",
    "annotations",
    "figures",
]


@dataclass
class Question:
    type: str
    difficulty: str
    case_data: Dict[str, Any]
    categories: List[str]
    sections: List[str]
    system_prompt: str

    def __post_init__(self) -> None:
        self.case_id: int = int(self.case_data["case_id"])
        self.case_content: str = self.select_case_sections()
        self.raw_content: Optional[str] = None
        self.content: Optional[Dict[str, Optional[str]]] = None

    def _annotation_summary(self) -> str:
        bboxes = self.case_data.get("bounding_boxes", []) or []
        if not bboxes:
            return "No bounding boxes provided."

        rows: List[str] = []
        for i, box in enumerate(bboxes, start=1):
            bbox = box.get("bbox") or {}
            rows.append(
                f"{i}. label={box.get('label', '')}; "
                f"x_min={bbox.get('x_min')}, y_min={bbox.get('y_min')}, "
                f"x_max={bbox.get('x_max')}, y_max={bbox.get('y_max')}"
            )
        return "\n".join(rows)

    def create_question_prompt(self) -> str:
        category_descriptions = "\n".join(
            f"- {name}: {desc}" for name, desc in CATEGORIES_META.items() if name in self.categories
        )

        return f"""
You must follow these rules:
1. Use only the provided testray case metadata and figure captions.
2. Do not invent objects, labels, coordinates, or risks not present in the case.
3. The question must explicitly mention referenced figure ids.
4. The answer must be objectively verifiable from provided fields.
5. Produce a difficult multiple-choice question with six options A-F.

Target capabilities:
{category_descriptions}

Case context:
{self.case_content}

Return exactly in this format:
THOUGHTS: [Reasoning plan for the agent]
QUESTION: [Full question with options A-F]
FIGURES: [List like ["Figure 1a"]]
EXPLANATION: [How the answer is verifiable from case metadata]
ANSWER: [Correct option letter, e.g. "C"]
""".strip()

    def select_case_sections(self) -> str:
        mapping = {
            "history": lambda c: c.get("history", "No history provided."),
            "image_finding": lambda c: c.get("image_finding", "No findings provided."),
            "diagnosis": lambda c: c.get("diagnosis", "No diagnosis provided."),
            "labels": lambda c: ", ".join(c.get("labels", [])) or "No labels provided.",
            "object_count": lambda c: str(c.get("object_count", "No object count provided.")),
            "features": lambda c: json.dumps(c.get("features", {}), ensure_ascii=False),
            "annotations": lambda c: self._annotation_summary(),
            "figures": self._format_figures,
        }

        formatted: List[str] = []
        for section in self.sections:
            fn = mapping.get(section)
            if fn is None:
                continue
            formatted.append(f"{section}:\n{fn(self.case_data)}")

        return "\n\n".join(formatted)

    @staticmethod
    def _format_figures(case_data: Dict[str, Any]) -> str:
        lines: List[str] = []
        for figure in case_data.get("figures", []):
            for subfig in figure.get("subfigures", []):
                lines.append(f"{subfig.get('number')}: {subfig.get('caption')}")
        return "\n".join(lines) if lines else "No figure captions provided."

    def create_question(
        self,
        client: openai.OpenAI,
        temperature: float = 0.4,
        top_p: float = 0.95,
        max_tokens: int = 1000,
        model: str = "qwen3-vl-plus",
    ) -> str:
        self.raw_content = get_llm_response(
            client=client,
            prompt=self.create_question_prompt(),
            system_prompt=self.system_prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            model=model,
        )
        self.content = self.extract_content()
        return self.raw_content

    def extract_content(self) -> Dict[str, Optional[str]]:
        if not self.raw_content:
            return {k: None for k in ["thoughts", "question", "figures", "explanation", "answer"]}

        keywords = ["THOUGHTS", "QUESTION", "FIGURES", "EXPLANATION", "ANSWER"]
        content: Dict[str, Optional[str]] = {}
        for kw in keywords:
            pattern = rf"{kw}:\s*(.*?)(?=\n[A-Z_]+:|$)"
            match = re.search(pattern, self.raw_content, re.DOTALL)
            content[kw.lower()] = match.group(1).strip() if match else None

        return content

    def save(self, output_path: str | Path) -> Dict[str, Any]:
        if self.content is None:
            raise RuntimeError("Question content is empty. Call create_question first.")

        output_dir = Path(output_path)
        case_dir = output_dir / str(self.case_id)
        case_dir.mkdir(parents=True, exist_ok=True)

        payload: Dict[str, Any] = dict(self.content)
        payload["metadata"] = {
            "case_id": self.case_id,
            "type": self.type,
            "difficulty": self.difficulty,
            "categories": self.categories,
            "sections": self.sections,
            "source_dataset": self.case_data.get("source_dataset", "testray"),
            "class_name": self.case_data.get("class_name"),
            "threat_level": self.case_data.get("threat_level"),
        }

        file_path = case_dir / f"{self.case_id}_{hash((self.case_id, tuple(self.categories), self.type, self.difficulty)) & 0xFFFFFFFF}.json"
        file_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload


def generate_questions(
    dataset: Dict[str, Any],
    client: openai.OpenAI,
    output_dir: str | Path,
    temperature: float = 0.4,
    top_p: float = 0.95,
    max_tokens: int = 1000,
    model: str = "qwen3-vl-plus",
    max_cases: Optional[int] = None,
) -> None:
    case_ids = sorted(dataset.keys(), key=lambda x: int(x))
    if max_cases is not None:
        case_ids = case_ids[:max_cases]

    for case_id in tqdm(case_ids, desc="Processing cases"):
        case_data = dataset[case_id]
        for combo in tqdm(CATEGORY_COMBINATIONS, desc=f"Categories for case {case_id}", leave=False):
            q = Question(
                type="multiple choice (A/B/C/D/E/F)",
                difficulty="complex",
                case_data=case_data,
                categories=combo,
                sections=DEFAULT_SECTIONS,
                system_prompt=SYSTEM_PROMPT,
            )
            q.create_question(
                client=client,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                model=model,
            )
            q.save(output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate baggage benchmark questions from testray metadata.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET_PATH, help="Path to testray_metadata.json")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR, help="Output directory for question JSON")
    parser.add_argument("--model", default="qwen3-vl-plus", help="LLM model name")
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=1000)
    parser.add_argument("--max_cases", type=int, default=None)
    parser.add_argument(
        "--threat_level",
        choices=["all", "threat", "non-threat"],
        default="all",
        help="Filter by threat level",
    )
    parser.add_argument(
        "--caption_keyword",
        action="append",
        default=None,
        help="Repeatable keyword filter for caption/image_finding",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    client = openai.OpenAI()

    dataset = load_testray_dataset(
        dataset_path=args.dataset,
        as_dict=True,
        threat_level=args.threat_level,
        caption_keywords=args.caption_keyword,
    )

    print(f"Loaded {len(dataset)} cases from {args.dataset}")
    if not dataset:
        print("No cases matched filters; exiting.")
        return

    generate_questions(
        dataset=dataset,
        client=client,
        output_dir=args.output,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        model=args.model,
        max_cases=args.max_cases,
    )

    print(f"Questions saved to: {args.output}")


if __name__ == "__main__":
    main()

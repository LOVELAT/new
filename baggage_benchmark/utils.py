from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


def _contains_keyword(text: str, keywords: Sequence[str]) -> bool:
    lower_text = text.lower()
    return any(k.lower() in lower_text for k in keywords)


def _match_caption_or_finding(case: Dict[str, Any], keywords: Sequence[str]) -> bool:
    if not keywords:
        return True

    if _contains_keyword(str(case.get("image_finding", "")), keywords):
        return True

    for figure in case.get("figures", []):
        for subfig in figure.get("subfigures", []):
            if _contains_keyword(str(subfig.get("caption", "")), keywords):
                return True

    return False


def _labels_match(case: Dict[str, Any], required_labels: Sequence[str]) -> bool:
    if not required_labels:
        return True

    case_labels = {str(x).strip().lower() for x in case.get("labels", [])}
    required = {str(x).strip().lower() for x in required_labels}
    return required.issubset(case_labels)


def load_testray_dataset(
    dataset_path: str | Path,
    as_dict: bool = False,
    threat_level: str = "all",
    class_names: Sequence[str] | None = None,
    required_labels: Sequence[str] | None = None,
    caption_keywords: Sequence[str] | None = None,
) -> List[Dict[str, Any]] | Dict[str, Dict[str, Any]]:
    """Load and filter the testray metadata dataset.

    Args:
        dataset_path: Path to `testray_metadata.json`.
        as_dict: Return keyed dict when True, otherwise list of case dicts.
        threat_level: One of `all`, `threat`, `non-threat`.
        class_names: Optional class-name allowlist.
        required_labels: Optional label subset that each case must include.
        caption_keywords: Optional keywords matched against captions/image_finding.

    Returns:
        Filtered dataset as dict or list.
    """

    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    # `utf-8-sig` gracefully handles files with or without UTF-8 BOM.
    with path.open("r", encoding="utf-8-sig") as f:
        data: Dict[str, Dict[str, Any]] = json.load(f)

    class_name_set = {x.strip().lower() for x in class_names} if class_names else set()
    keyword_list = list(caption_keywords or [])
    label_list = list(required_labels or [])

    filtered: Dict[str, Dict[str, Any]] = {}
    for case_id, case in data.items():
        case_threat = str(case.get("threat_level", "")).strip().lower()
        if threat_level != "all" and case_threat != threat_level:
            continue

        if class_name_set:
            this_class = str(case.get("class_name", "")).strip().lower()
            if this_class not in class_name_set:
                continue

        if not _labels_match(case, label_list):
            continue

        if not _match_caption_or_finding(case, keyword_list):
            continue

        filtered[case_id] = case

    if as_dict:
        return filtered

    return list(filtered.values())


def save_dataset(dataset: Dict[str, Any] | List[Dict[str, Any]], dataset_path: str | Path) -> None:
    """Save a dataset to JSON with UTF-8 encoding."""

    path = Path(dataset_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

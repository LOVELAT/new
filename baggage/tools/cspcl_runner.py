from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

RUNNER_JSON_PREFIX = "__CSPCL_JSON__="

DATASET_CLASS_MAPPINGS: Dict[str, Tuple[str, ...]] = {
    "pidray_detection.py": (
        "Gun",
        "Bullet",
        "Knife",
        "Wrench",
        "Pliers",
        "Powerbank",
        "Baton",
        "Lighter",
        "Sprayer",
        "Hammer",
        "Scissors",
        "HandCuffs",
    ),
    "pixray_detection.py": (
        "Gun",
        "Knife",
        "Lighter",
        "Battery",
        "Pliers",
        "Scissors",
        "Wrench",
        "Hammer",
        "Screwdriver",
        "Dart",
        "Bat",
        "Fireworks",
        "Saw_blade",
        "Razor_blade",
        "Pressure_vessel",
    ),
    "clcxray_detection.py": (
        "blade",
        "scissors",
        "knife",
        "dagger",
        "SwissArmyKnife",
        "PlasticBottle",
        "Cans",
        "VacuumCup",
        "GlassBottle",
        "CartonDrinks",
        "Tin",
        "SprayCans",
    ),
}


def _get_mmdet_api(name: str) -> Any:
    module = importlib.import_module("mmdet.apis")
    return getattr(module, name)


def _get_mmdet_util(name: str) -> Any:
    module = importlib.import_module("mmdet.utils")
    return getattr(module, name)


def _resolve_class_names(
    model: Any,
    config_path: Path,
    override_names: Optional[Sequence[str]] = None,
) -> Sequence[str]:
    if override_names:
        return tuple(override_names)

    dataset_meta = getattr(model, "dataset_meta", None) or {}
    if isinstance(dataset_meta, dict):
        classes = dataset_meta.get("classes")
        if classes:
            return tuple(classes)

    try:
        cfg_text = config_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        cfg_text = ""

    for dataset_marker, classes in DATASET_CLASS_MAPPINGS.items():
        if dataset_marker in cfg_text:
            return classes

    return tuple(f"class_{idx}" for idx in range(100))


def _predict(
    image_path: Path,
    config_path: Path,
    checkpoint_path: Path,
    score_threshold: float,
    device: str,
    class_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    register_all_modules = _get_mmdet_util("register_all_modules")
    try:
        register_all_modules(init_default_scope=False)
    except TypeError:
        register_all_modules()

    init_detector = _get_mmdet_api("init_detector")
    inference_detector = _get_mmdet_api("inference_detector")

    model = init_detector(str(config_path), str(checkpoint_path), device=device)
    result = inference_detector(model, str(image_path))
    pred_instances = result.pred_instances

    scores = pred_instances.scores.tolist()
    labels = pred_instances.labels.tolist()
    bboxes = pred_instances.bboxes.tolist()
    resolved_class_names = list(_resolve_class_names(model, config_path, class_names))

    detections: List[Dict[str, Any]] = []
    for score, label, bbox in zip(scores, labels, bboxes):
        if float(score) < score_threshold:
            continue

        label_index = int(label)
        label_name = (
            resolved_class_names[label_index]
            if 0 <= label_index < len(resolved_class_names)
            else f"class_{label_index}"
        )
        detections.append(
            {
                "label": label_name,
                "label_id": label_index,
                "score": round(float(score), 6),
                "bbox": {
                    "x_min": round(float(bbox[0]), 3),
                    "y_min": round(float(bbox[1]), 3),
                    "x_max": round(float(bbox[2]), 3),
                    "y_max": round(float(bbox[3]), 3),
                },
            }
        )

    detections.sort(key=lambda item: item["score"], reverse=True)
    return {
        "image_path": str(image_path),
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "device": device,
        "score_threshold": score_threshold,
        "num_detections": len(detections),
        "detections": detections,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--class-names-json")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
        importlib.invalidate_caches()

    override_names = json.loads(args.class_names_json) if args.class_names_json else None

    try:
        payload = _predict(
            image_path=Path(args.image_path).resolve(),
            config_path=Path(args.config_path).resolve(),
            checkpoint_path=Path(args.checkpoint_path).resolve(),
            score_threshold=args.score_threshold,
            device=args.device,
            class_names=override_names,
        )
    except Exception as exc:
        payload = {
            "error": f"CSPCL inference failed: {exc}",
            "image_path": str(Path(args.image_path).resolve()),
            "config_path": str(Path(args.config_path).resolve()),
            "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
            "device": args.device,
        }
        print(f"{RUNNER_JSON_PREFIX}{json.dumps(payload, ensure_ascii=True)}")
        return 1

    print(f"{RUNNER_JSON_PREFIX}{json.dumps(payload, ensure_ascii=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

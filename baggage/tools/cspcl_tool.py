from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

RUNNER_JSON_PREFIX = "__CSPCL_JSON__="
RUNNER_LOG_PREVIEW_CHARS = 1200


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


class CSPCLDetectorInput(BaseModel):
    image_path: str = Field(..., description="Path to a local X-ray image.")
    config_path: Optional[str] = Field(
        default=None,
        description="Path to a CSPCL config file. Can be absolute or relative to the CSPCL repo root.",
    )
    checkpoint_path: Optional[str] = Field(
        default=None,
        description="Path to a model checkpoint (.pth). Can be absolute or relative to the CSPCL repo root.",
    )
    score_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Only keep detections with score >= this threshold.",
    )
    device: Optional[str] = Field(
        default=None,
        description="Inference device such as 'cpu' or 'cuda:0'. Defaults to the tool's configured device.",
    )
    return_visualization: bool = Field(
        default=True,
        description="Whether to save and return a visualization image with predicted boxes.",
    )
    visualization_path: Optional[str] = Field(
        default=None,
        description="Optional explicit path for the visualization image output.",
    )
    save_json: bool = Field(
        default=True,
        description="Whether to save the structured detection result to a JSON file.",
    )
    json_output_path: Optional[str] = Field(
        default=None,
        description="Optional explicit path for the JSON output file.",
    )
    class_names: Optional[List[str]] = Field(
        default=None,
        description="Optional class names override when the checkpoint does not contain dataset metadata.",
    )


class CSPCLDetectorTool(BaseTool):
    name: str = "cspcl_detector"
    description: str = (
        "Run a CSPCL/MMDetection prohibited-item detector on a baggage X-ray image. "
        "Input should include image_path, config_path, and checkpoint_path unless defaults are configured."
    )
    args_schema: type[BaseModel] = CSPCLDetectorInput

    _repo_root: Path = PrivateAttr()
    _temp_dir: Path = PrivateAttr()
    _default_device: str = PrivateAttr()
    _default_config: Optional[Path] = PrivateAttr(default=None)
    _default_checkpoint: Optional[Path] = PrivateAttr(default=None)
    _model_cache: Dict[Tuple[str, str, str], Any] = PrivateAttr(default_factory=dict)
    _external_python: Optional[Path] = PrivateAttr(default=None)
    _runner_script: Path = PrivateAttr()

    def __init__(
        self,
        repo_root: str | Path,
        temp_dir: str | Path,
        device: str = "cpu",
        default_config: str | Path | None = None,
        default_checkpoint: str | Path | None = None,
        python_path: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._repo_root = Path(repo_root).resolve()
        self._temp_dir = Path(temp_dir).resolve()
        self._temp_dir.mkdir(parents=True, exist_ok=True)
        self._default_device = device
        self._default_config = Path(default_config).resolve() if default_config else None
        self._default_checkpoint = Path(default_checkpoint).resolve() if default_checkpoint else None
        self._external_python = Path(python_path).resolve() if python_path else None
        self._runner_script = Path(__file__).resolve().with_name("cspcl_runner.py")

    def _run(
        self,
        image_path: str,
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        score_threshold: float = 0.3,
        device: Optional[str] = None,
        return_visualization: bool = True,
        visualization_path: Optional[str] = None,
        save_json: bool = True,
        json_output_path: Optional[str] = None,
        class_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        image = Path(image_path).expanduser().resolve()
        if not image.exists():
            return {"error": f"Image not found: {image}"}

        try:
            config = self._resolve_config_path(config_path)
            checkpoint = self._resolve_checkpoint_path(checkpoint_path)
        except FileNotFoundError as exc:
            return {"error": str(exc)}
        except ValueError as exc:
            return {"error": str(exc)}

        inference_device = device or self._default_device

        if self._external_python is not None:
            output = self._run_external(
                image_path=image,
                config_path=config,
                checkpoint_path=checkpoint,
                score_threshold=score_threshold,
                device=inference_device,
                class_names=class_names,
            )
        else:
            try:
                model = self._load_model(config=config, checkpoint=checkpoint, device=inference_device)
                detections = self._predict(
                    model=model,
                    image_path=image,
                    score_threshold=score_threshold,
                    class_names=class_names,
                )
            except Exception as exc:
                return {
                    "error": f"CSPCL inference failed: {exc}",
                    "image_path": str(image),
                    "config_path": str(config),
                    "checkpoint_path": str(checkpoint),
                    "device": inference_device,
                }

            output = {
                "image_path": str(image),
                "config_path": str(config),
                "checkpoint_path": str(checkpoint),
                "device": inference_device,
                "score_threshold": score_threshold,
                "num_detections": len(detections),
                "detections": detections,
            }

        if return_visualization:
            vis_path = self._resolve_visualization_path(image, visualization_path)
            self._save_visualization(image, output.get("detections", []), vis_path)
            output["visualization_path"] = str(vis_path)

        if save_json:
            json_path = self._resolve_json_output_path(image, json_output_path)
            self._save_json_output(output, json_path)
            output["json_output_path"] = str(json_path)

        return output

    async def _arun(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return self._run(*args, **kwargs)

    def _resolve_config_path(self, config_path: Optional[str]) -> Path:
        candidates: List[Path] = []
        if config_path:
            candidates.append(Path(config_path))
        elif self._default_config is not None:
            candidates.append(self._default_config)
        elif os.getenv("CSPCL_CONFIG_PATH"):
            candidates.append(Path(os.environ["CSPCL_CONFIG_PATH"]))
        else:
            raise ValueError(
                "No config_path provided. Set it explicitly or configure CSPCL_CONFIG_PATH."
            )

        for candidate in candidates:
            resolved = self._resolve_under_repo(candidate)
            if resolved.exists():
                return resolved

        raise FileNotFoundError(f"CSPCL config not found: {candidates[0]}")

    def _resolve_checkpoint_path(self, checkpoint_path: Optional[str]) -> Path:
        candidates: List[Path] = []
        if checkpoint_path:
            candidates.append(Path(checkpoint_path))
        elif self._default_checkpoint is not None:
            candidates.append(self._default_checkpoint)
        elif os.getenv("CSPCL_CHECKPOINT_PATH"):
            candidates.append(Path(os.environ["CSPCL_CHECKPOINT_PATH"]))
        else:
            raise ValueError(
                "No checkpoint_path provided. Set it explicitly or configure CSPCL_CHECKPOINT_PATH."
            )

        for candidate in candidates:
            resolved = self._resolve_under_repo(candidate)
            if resolved.exists():
                return resolved

        raise FileNotFoundError(f"CSPCL checkpoint not found: {candidates[0]}")

    def _resolve_under_repo(self, path_value: Path) -> Path:
        if path_value.is_absolute():
            return path_value.expanduser().resolve()

        repo_candidate = (self._repo_root / path_value).resolve()
        if repo_candidate.exists():
            return repo_candidate

        workspace_candidate = (Path.cwd() / path_value).resolve()
        return workspace_candidate

    def _run_external(
        self,
        image_path: Path,
        config_path: Path,
        checkpoint_path: Path,
        score_threshold: float,
        device: str,
        class_names: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        if self._external_python is None:
            raise RuntimeError("External CSPCL python interpreter is not configured.")
        if not self._external_python.exists():
            return {"error": f"CSPCL python not found: {self._external_python}"}
        if not self._runner_script.exists():
            return {"error": f"CSPCL runner script not found: {self._runner_script}"}

        command = [
            str(self._external_python),
            str(self._runner_script),
            "--repo-root",
            str(self._repo_root),
            "--image-path",
            str(image_path),
            "--config-path",
            str(config_path),
            "--checkpoint-path",
            str(checkpoint_path),
            "--score-threshold",
            str(score_threshold),
            "--device",
            device,
        ]
        if class_names:
            command.extend(["--class-names-json", json.dumps(list(class_names), ensure_ascii=True)])

        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(Path(__file__).resolve().parents[2]),
            check=False,
        )

        payload: Optional[Dict[str, Any]] = None
        stdout_lines = completed.stdout.splitlines()
        for line in reversed(stdout_lines):
            if line.startswith(RUNNER_JSON_PREFIX):
                payload = json.loads(line[len(RUNNER_JSON_PREFIX) :])
                break


         #completed.stdout 是 runner 的标准输出
         # splitlines() 把它拆成多行
         # 倒序遍历每一行
         # 找到形如__CSPCL_JSON__={...}的那一行
         # 去掉前缀 RUNNER_JSON_PREFIX
         # 对后面的 JSON 字符串做 json.loads(...)
         # 得到 Python 字典 payload



        runner_logs = "\n".join(
            line for line in stdout_lines if not line.startswith(RUNNER_JSON_PREFIX)
        ).strip()
        stderr_logs = completed.stderr.strip()

        # 把普通日志和错误日志单独提出来

        if payload is None:
            details = "\n".join(part for part in (runner_logs, stderr_logs) if part)
            return {
                "error": "CSPCL external runner did not return structured output.",
                "image_path": str(image_path),
                "config_path": str(config_path),
                "checkpoint_path": str(checkpoint_path),
                "device": device,
                "runner_logs": details,
                "return_code": completed.returncode,
            }
        
        # 如果根本没拿到 JSON，就返回错误

        if runner_logs:
            payload["runner_stdout"] = self._truncate_log(runner_logs)
            if "do not match exactly" in runner_logs:
                payload["model_warning"] = (
                    "Checkpoint and config do not match exactly. Inference still ran, "
                    "but the current config may not be the exact training config."
                )
        if stderr_logs and completed.returncode != 0:
            payload["runner_stderr"] = self._truncate_log(stderr_logs)
        payload["return_code"] = completed.returncode
        return payload
    
         # payload["runner_stdout"]
         # 保存 stdout 里的普通日志，但会先截断，避免太长
         # model_warning
         # 如果日志里出现 "do not match exactly"，说明权重和 config 不完全匹配
         # 于是额外加一个更友好的警告字段
         # payload["runner_stderr"]
         # 如果 stderr 有内容且进程返回码不是 0，就把错误日志也加进去
         # payload["return_code"]
         # 记录子进程退出码，0 通常表示成功

    @staticmethod
    def _truncate_log(text: str) -> str:
        if len(text) <= RUNNER_LOG_PREVIEW_CHARS:
            return text
        return f"{text[:RUNNER_LOG_PREVIEW_CHARS]}...(truncated)"

    def _load_model(self, config: Path, checkpoint: Path, device: str) -> Any:
        cache_key = (str(config), str(checkpoint), device)
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        self._prepare_cspcl_imports()
        init_detector = self._get_mmdet_api("init_detector")

        model = init_detector(str(config), str(checkpoint), device=device)
        self._model_cache[cache_key] = model
        return model

    def _prepare_cspcl_imports(self) -> None:
        repo_root_str = str(self._repo_root)
        if repo_root_str not in sys.path:
            sys.path.insert(0, repo_root_str)
            importlib.invalidate_caches()

        register_all_modules = self._get_mmdet_util("register_all_modules")

        try:
            register_all_modules(init_default_scope=False)
        except TypeError:
            register_all_modules()

    def _predict(
        self,
        model: Any,
        image_path: Path,
        score_threshold: float,
        class_names: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        inference_detector = self._get_mmdet_api("inference_detector")

        result = inference_detector(model, str(image_path))
        pred_instances = result.pred_instances

        scores = pred_instances.scores.tolist()
        labels = pred_instances.labels.tolist()
        bboxes = pred_instances.bboxes.tolist()
        resolved_class_names = list(class_names or self._resolve_class_names(model, image_path))

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
        return detections

    def _resolve_class_names(self, model: Any, image_path: Path) -> Sequence[str]:
        dataset_meta = getattr(model, "dataset_meta", None) or {}
        if isinstance(dataset_meta, dict):
            classes = dataset_meta.get("classes")
            if classes:
                return tuple(classes)

        cfg_text = ""
        model_cfg = getattr(model, "cfg", None)
        if model_cfg is not None and getattr(model_cfg, "filename", None):
            try:
                cfg_text = Path(model_cfg.filename).read_text(encoding="utf-8", errors="ignore")
            except OSError:
                cfg_text = ""

        for dataset_marker, classes in DATASET_CLASS_MAPPINGS.items():
            if dataset_marker in cfg_text:
                return classes

        return tuple(f"class_{idx}" for idx in range(100))

    def _resolve_visualization_path(self, image_path: Path, explicit_path: Optional[str]) -> Path:
        if explicit_path:
            path = Path(explicit_path).expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            return path

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        return self._temp_dir / f"{image_path.stem}_cspcl_{timestamp}{image_path.suffix}"

    def _resolve_json_output_path(self, image_path: Path, explicit_path: Optional[str]) -> Path:
        if explicit_path:
            path = Path(explicit_path).expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            return path

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        return self._temp_dir / f"{image_path.stem}_cspcl_{timestamp}.json"

    @staticmethod
    def _save_visualization(image_path: Path, detections: Sequence[Dict[str, Any]], output_path: Path) -> None:
        from PIL import Image, ImageDraw

        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        for det in detections:
            bbox = det["bbox"]
            label = det["label"]
            score = det["score"]
            box = [bbox["x_min"], bbox["y_min"], bbox["x_max"], bbox["y_max"]]
            draw.rectangle(box, outline="red", width=3)
            draw.text((bbox["x_min"] + 4, bbox["y_min"] + 4), f"{label} {score:.2f}", fill="yellow")

        image.save(output_path)

    @staticmethod
    def _get_mmdet_api(name: str) -> Any:
        module = importlib.import_module("mmdet.apis")
        return getattr(module, name)

    @staticmethod
    def _get_mmdet_util(name: str) -> Any:
        module = importlib.import_module("mmdet.utils")
        return getattr(module, name)


def create_cspcl_detector_tool(
    temp_dir: str = "temp",
    device: str = "cpu",
    repo_root: str | Path | None = None,
    default_config: str | Path | None = None,
    default_checkpoint: str | Path | None = None,
    python_path: str | Path | None = None,
    **_: Any,
) -> CSPCLDetectorTool:
    workspace_root = Path(__file__).resolve().parents[2]
    resolved_repo_root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else (workspace_root / "CSPCL" / "CSPCL").resolve()
    )

    fallback_config = resolved_repo_root / "configs" / "dino" / "AO-DETR-BASE_r50_pixray.py"
    fallback_checkpoint = workspace_root / "CSPCL" / "AO-DETR_r50_PIXray320.pth"
    env_python_path = os.getenv("CSPCL_PYTHON_PATH")
    env_default_config = os.getenv("CSPCL_CONFIG_PATH")
    env_default_checkpoint = os.getenv("CSPCL_CHECKPOINT_PATH")
    detected_python = (
        Path(python_path).resolve()
        if python_path is not None
        else Path(env_python_path).resolve()
        if env_python_path
        else (Path("F:/Anaconda/envs/cspcl39/python.exe").resolve())
    )

#解释器优先级
# 代码里显式传的 python_path
# 环境变量 CSPCL_PYTHON_PATH
# 默认值 F:\Anaconda\envs\cspcl39\python.exe


    return CSPCLDetectorTool(
        repo_root=resolved_repo_root,
        temp_dir=temp_dir,
        device=device,
        default_config=default_config or env_default_config or (fallback_config if fallback_config.exists() else None),
        default_checkpoint=default_checkpoint
        or env_default_checkpoint
        or (str(fallback_checkpoint) if fallback_checkpoint.exists() else None),
        python_path=str(detected_python) if detected_python.exists() else None,
    )


__all__ = ["CSPCLDetectorTool", "create_cspcl_detector_tool"]

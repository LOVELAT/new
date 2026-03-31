#!/usr/bin/env python3
"""Build Testray metadata JSON in a Eurorad-like schema.

Usage:
    python generate_testray_metadata.py --root "F:/Project/new/testray"
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import xml.etree.ElementTree as ET

NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _cell_text(cell: ET.Element, shared_strings: List[str]) -> Optional[str]:
    cell_type = cell.attrib.get("t")

    if cell_type == "s":
        value_node = cell.find("x:v", NS)
        if value_node is None or value_node.text is None:
            return None
        idx = int(value_node.text)
        if 0 <= idx < len(shared_strings):
            return shared_strings[idx]
        return None

    if cell_type == "inlineStr":
        return "".join((t.text or "") for t in cell.findall(".//x:t", NS))

    raw = cell.find("x:v", NS)
    return None if raw is None else raw.text


def parse_xlsx_rows(xlsx_path: Path) -> List[Dict[str, Optional[str]]]:
    with zipfile.ZipFile(xlsx_path) as zf:
        shared_strings: List[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            shared_root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in shared_root.findall("x:si", NS):
                shared_strings.append("".join((t.text or "") for t in si.findall(".//x:t", NS)))

        sheet_root = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))

    rows: List[Dict[str, Optional[str]]] = []
    for row in sheet_root.findall(".//x:sheetData/x:row", NS):
        values: Dict[str, Optional[str]] = {}
        for cell in row.findall("x:c", NS):
            ref = cell.attrib.get("r", "")
            match = re.match(r"^[A-Z]+", ref)
            if not match:
                continue
            col = match.group(0)
            values[col] = _cell_text(cell, shared_strings)

        rows.append(
            {
                "row": row.attrib.get("r"),
                "A": values.get("A"),
                "B": values.get("B"),
                "C": values.get("C"),
                "D": values.get("D"),
            }
        )

    return rows


def parse_class_info(class_folder: str) -> Dict[str, Any]:
    m = re.match(r"^Class\s+(\d+)_(.+)$", class_folder)
    if m:
        return {"class_id": int(m.group(1)), "class_name": m.group(2).strip()}

    m = re.match(r"^Class\s+(\d+)$", class_folder)
    if m:
        return {"class_id": int(m.group(1)), "class_name": class_folder}

    return {"class_id": None, "class_name": class_folder}


def parse_image_tokens(stem: str) -> Dict[str, Any]:
    parts = stem.split("_")
    result: Dict[str, Any] = {
        "object_code": parts[0] if parts else None,
        "bag": None,
        "layer": None,
        "camera": None,
        "location": None,
        "phi": None,
        "theta": None,
        "frame": None,
        "raw_tokens": parts,
    }

    for part in parts:
        if re.match(r"^B\d+$", part):
            result["bag"] = part
            continue
        if re.match(r"^Loc\d+$", part):
            result["location"] = part
            continue
        if re.match(r"^L(?!oc).+$", part):
            result["layer"] = part
            continue
        if re.match(r"^C\d+$", part):
            result["camera"] = part
            continue
        if re.match(r"^phi-?\d+$", part):
            result["phi"] = part
            continue
        if re.match(r"^th-?\d+$", part):
            result["theta"] = part
            continue

    if parts and re.match(r"^\d+$", parts[-1]):
        result["frame"] = int(parts[-1])

    return result


def split_frame_suffix(stem: str) -> tuple[str, Optional[int]]:
    m = re.match(r"^(.*)_(\d+)$", stem)
    if not m:
        return stem, None
    return m.group(1), int(m.group(2))


def subfigure_number(index: int) -> str:
    # Typical groups have <= 4 views; fallback keeps deterministic ids if larger.
    if 0 <= index < 26:
        return f"Figure 1{chr(ord('a') + index)}"
    return f"Figure 1v{index + 1}"


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def shape_summary(shape: Dict[str, Any]) -> Dict[str, Any]:
    points_xy: List[Dict[str, float]] = []
    raw_points = shape.get("points") or []

    for pt in raw_points:
        if not isinstance(pt, list) or len(pt) < 2:
            continue
        points_xy.append({"x": float(pt[0]), "y": float(pt[1])})

    bbox = None
    if points_xy:
        xs = [p["x"] for p in points_xy]
        ys = [p["y"] for p in points_xy]
        bbox = {
            "x_min": min(xs),
            "y_min": min(ys),
            "x_max": max(xs),
            "y_max": max(ys),
            "width": max(xs) - min(xs),
            "height": max(ys) - min(ys),
        }

    return {
        "label": str(shape.get("label", "")),
        "shape_type": str(shape.get("shape_type", "")),
        "point_count": len(points_xy),
        "points": points_xy,
        "bbox": bbox,
    }


def build_metadata(root: Path) -> Dict[str, Dict[str, Any]]:
    captions_by_class: Dict[str, Dict[str, str]] = {}
    for xlsx in sorted((root / "Captions").glob("*.xlsx")):
        class_folder = xlsx.stem
        caption_map: Dict[str, str] = {}
        for row in parse_xlsx_rows(xlsx):
            row_num = int(row.get("row") or 0)
            image_name = row.get("A")
            caption = row.get("B")
            if row_num <= 1 or not image_name:
                continue
            caption_map[image_name] = caption or ""
        captions_by_class[class_folder] = caption_map

    image_files = sorted(
        [
            p
            for p in (root / "Images").rglob("*")
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
    )

    grouped_images: Dict[tuple[str, str], List[Path]] = {}
    for image_path in image_files:
        class_folder = image_path.parent.name
        base_stem, _ = split_frame_suffix(image_path.stem)
        grouped_images.setdefault((class_folder, base_stem), []).append(image_path)

    def group_sort_key(item: tuple[tuple[str, str], List[Path]]) -> tuple[Any, str, str]:
        class_folder, base_stem = item[0]
        class_info = parse_class_info(class_folder)
        class_id = class_info["class_id"] if isinstance(class_info["class_id"], int) else 10**9
        return class_id, class_folder.lower(), base_stem.lower()

    dataset: Dict[str, Dict[str, Any]] = {}
    case_id = 1

    for (class_folder, base_stem), views in sorted(grouped_images.items(), key=group_sort_key):
        class_info = parse_class_info(class_folder)
        threat_level = "non-threat" if "Non Threat" in class_folder else "threat"

        figures: List[Dict[str, str]] = []
        labels_set = set()
        per_view_object_counts: Dict[str, int] = {}
        image_sizes: Dict[str, Dict[str, Any]] = {}
        annotation_views: List[Dict[str, Optional[str]]] = []
        group_polygons: List[Dict[str, Any]] = []
        group_bboxes: List[Dict[str, Any]] = []
        view_tokens: List[Dict[str, Any]] = []
        view_frames: List[int] = []

        def view_sort_key(path: Path) -> tuple[Any, str]:
            _, frame = split_frame_suffix(path.stem)
            frame_order = frame if frame is not None else 10**9
            return frame_order, path.name.lower()

        for idx, image_path in enumerate(sorted(views, key=view_sort_key)):
            stem = image_path.stem
            tokens = parse_image_tokens(stem)
            frame = tokens.get("frame")
            if isinstance(frame, int):
                view_frames.append(frame)

            polygon_path = root / "Json" / class_folder / f"{stem}.json"
            bbox_path = root / "Json_BB" / class_folder / f"{stem}.json"
            seg_path = root / "Segmentation" / class_folder / f"{stem}.png"

            polygon_data = load_json(polygon_path)
            bbox_data = load_json(bbox_path)

            polygon_shapes = [shape_summary(s) for s in (polygon_data or {}).get("shapes", [])]
            bbox_shapes = [shape_summary(s) for s in (bbox_data or {}).get("shapes", [])]

            view_labels = {s["label"] for s in (polygon_shapes + bbox_shapes) if s.get("label")}
            labels_set.update(view_labels)

            view_object_count = len(polygon_shapes) if polygon_shapes else len(bbox_shapes)
            per_view_object_counts[image_path.name] = view_object_count

            image_width = (polygon_data or {}).get("imageWidth")
            image_height = (polygon_data or {}).get("imageHeight")
            if image_width is None:
                image_width = (bbox_data or {}).get("imageWidth")
            if image_height is None:
                image_height = (bbox_data or {}).get("imageHeight")
            image_sizes[image_path.name] = {
                "width": image_width,
                "height": image_height,
            }

            caption = captions_by_class.get(class_folder, {}).get(image_path.name) or "No caption provided."
            rel_image = image_path.relative_to(root).as_posix()
            rel_poly = polygon_path.relative_to(root).as_posix() if polygon_path.exists() else None
            rel_bbox = bbox_path.relative_to(root).as_posix() if bbox_path.exists() else None
            rel_seg = seg_path.relative_to(root).as_posix() if seg_path.exists() else None

            figure_id = subfigure_number(idx)
            figures.append(
                {
                    "number": figure_id,
                    "url": rel_image,
                    "caption": caption,
                }
            )

            annotation_views.append(
                {
                    "image_filename": image_path.name,
                    "figure_number": figure_id,
                    "polygon_json": rel_poly,
                    "bbox_json": rel_bbox,
                    "segmentation_mask": rel_seg,
                }
            )

            for shape in polygon_shapes:
                enriched = dict(shape)
                enriched["figure_number"] = figure_id
                enriched["image_filename"] = image_path.name
                group_polygons.append(enriched)

            for shape in bbox_shapes:
                enriched = dict(shape)
                enriched["figure_number"] = figure_id
                enriched["image_filename"] = image_path.name
                group_bboxes.append(enriched)

            view_tokens.append(
                {
                    "image_filename": image_path.name,
                    "tokens": tokens,
                }
            )

        labels = sorted(labels_set)
        object_count = max(per_view_object_counts.values()) if per_view_object_counts else 0
        total_observations = sum(per_view_object_counts.values())
        diagnosis = ", ".join(labels) if labels else str(class_info["class_name"])
        label_text = ", ".join(labels) if labels else "none"
        unique_counts = sorted(set(per_view_object_counts.values()))
        if not unique_counts:
            per_view_count_text = "no annotated objects in available views"
        elif len(unique_counts) == 1:
            per_view_count_text = f"{unique_counts[0]} annotated object(s) per view"
        else:
            per_view_count_text = (
                f"{unique_counts[0]}-{unique_counts[-1]} annotated object(s) per view"
            )

        image_finding = (
            f"Security X-ray case with {len(figures)} view(s) of the same baggage sample; "
            f"{per_view_count_text}; labels: {label_text}."
        )

        group_tokens = parse_image_tokens(base_stem)
        group_tokens["view_frames"] = sorted(set(view_frames))
        group_tokens["view_count"] = len(figures)
        group_tokens["group_stem"] = base_stem
        group_tokens["per_view_tokens"] = view_tokens

        first_view = annotation_views[0] if annotation_views else {}
        first_image_filename = str(first_view.get("image_filename", "")) if first_view else ""
        first_image_size = image_sizes.get(first_image_filename, {"width": None, "height": None})

        dataset[str(case_id)] = {
            "case_id": case_id,
            "title": f"{class_folder} - {base_stem}",
            "section": "Security screening X-ray",
            "diagnosis": diagnosis,
            "history": (
                f"Testray sample from class '{class_folder}' with threat level '{threat_level}', "
                f"containing {len(figures)} X-ray view(s) of one baggage case."
            ),
            "image_finding": image_finding,
            "discussion": (
                "Features include grouped multi-view class metadata, parsed filename tokens, polygon annotations, "
                "bounding boxes, segmentation mask paths, and caption text."
            ),
            "differential_diagnosis": str(class_info["class_name"]),
            "figures": [
                {
                    "number": "Figure 1",
                    "subfigures": figures,
                }
            ],
            "area_of_interest": ["Security Screening", "Threat Detection"],
            "imaging_technique": ["X-ray"],
            "link": "",
            "time": datetime.now().strftime("%d.%m.%Y"),
            "source_dataset": "testray",
            "class_id": class_info["class_id"],
            "class_name": class_info["class_name"],
            "threat_level": threat_level,
            "image_filename": first_image_filename,
            "image_filenames": [str(v.get("image_filename", "")) for v in annotation_views],
            "image_size": first_image_size,
            "image_sizes": image_sizes,
            "annotation_files": {
                "polygon_json": first_view.get("polygon_json"),
                "bbox_json": first_view.get("bbox_json"),
                "segmentation_mask": first_view.get("segmentation_mask"),
                "all_views": annotation_views,
            },
            "labels": labels,
            "object_count": object_count,
            "object_count_per_view": per_view_object_counts,
            "object_count_total_observations": total_observations,
            "features": group_tokens,
            "polygons": group_polygons,
            "bounding_boxes": group_bboxes,
        }

        case_id += 1

    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate testray metadata JSON.")
    parser.add_argument("--root", default=r"F:\Project\new\testray", help="Testray root folder")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    root = Path(args.root)
    output = Path(args.output) if args.output else root / "testray_metadata.json"

    dataset = build_metadata(root)
    output.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Generated metadata for {len(dataset)} grouped cases.")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()

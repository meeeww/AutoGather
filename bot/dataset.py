from __future__ import annotations

import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any

import cv2
import yaml

from bot.config import ROOT_DIR, SECRETS_PATH

ROBOFLOW_WORKSPACE = "albiononline-c8fxi"
ROBOFLOW_PROJECT = "albiongathering"
ROBOFLOW_VERSION = 4
DATASET_URL = "https://universe.roboflow.com/albiononline-c8fxi/albiongathering/dataset/4"

DATASET_DIR = ROOT_DIR / "dataset"
YOLO_DIR = DATASET_DIR / "yolo"
CROPS_DIR = DATASET_DIR / "crops"
MANIFEST_PATH = DATASET_DIR / "manifest.json"
CUSTOM_DIR = ROOT_DIR / "custom_items"
CUSTOM_MANIFEST_PATH = CUSTOM_DIR / "manifest.json"

SKIP_CLASSES = {"monster"}
GATHER_CLASSES = [
    "granite",
    "iron ore",
    "ressources",
    "rought log",
    "sandstone",
    "tin ore",
    "titanium",
    "travertine",
]
DISPLAY_NAMES = {
    "granite": "Granite",
    "iron ore": "Iron Ore",
    "ressources": "Resources",
    "rought log": "Rough Log",
    "sandstone": "Sandstone",
    "tin ore": "Tin Ore",
    "titanium": "Titanium",
    "travertine": "Travertine",
}
MAX_CROPS_PER_CLASS = 8
MIN_BOX_PX = 20


def class_slug(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def display_name(class_name: str) -> str:
    return DISPLAY_NAMES.get(class_name, class_name.replace("_", " ").title())


def load_api_key() -> str:
    env_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if env_key:
        return env_key
    if SECRETS_PATH.exists():
        with SECRETS_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return str(data.get("roboflow_api_key", "")).strip()
    return ""


def save_api_key(api_key: str) -> None:
    api_key = api_key.strip()
    payload: dict[str, str] = {}
    if SECRETS_PATH.exists():
        with SECRETS_PATH.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    if api_key:
        payload["roboflow_api_key"] = api_key
    else:
        payload.pop("roboflow_api_key", None)
    if payload:
        with SECRETS_PATH.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
    elif SECRETS_PATH.exists():
        SECRETS_PATH.unlink()


def download_roboflow(api_key: str, dest: Path = YOLO_DIR) -> Path:
    if not api_key.strip():
        raise ValueError("A Roboflow API key is required to download the dataset.")
    from roboflow import Roboflow

    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    rf = Roboflow(api_key=api_key.strip())
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    dataset = project.version(ROBOFLOW_VERSION).download(
        "yolov8",
        location=str(dest),
        overwrite=True,
    )
    return Path(getattr(dataset, "location", dest))


def import_zip(zip_path: Path, dest: Path = YOLO_DIR) -> Path:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(dest)
    return dest


def import_folder(source: Path, dest: Path = YOLO_DIR) -> Path:
    yaml_path = find_data_yaml(source)
    root = yaml_path.parent
    if dest.resolve() != root.resolve():
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(root, dest)
        return dest
    return root


def find_data_yaml(root: Path) -> Path:
    matches = list(root.rglob("data.yaml"))
    if not matches:
        raise FileNotFoundError(f"No data.yaml found under {root}")
    matches.sort(key=lambda path: len(path.parts))
    return matches[0]


def _parse_names(raw: Any) -> list[str]:
    if isinstance(raw, dict):
        return [str(raw[key]) for key in sorted(raw, key=lambda item: int(item))]
    if isinstance(raw, list):
        return [str(item) for item in raw]
    raise ValueError("Could not read class names from data.yaml")


def _iter_yolo_pairs(yolo_root: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    split_found = False
    for split in ("train", "valid", "test"):
        image_dir = yolo_root / split / "images"
        label_dir = yolo_root / split / "labels"
        if not image_dir.exists():
            continue
        split_found = True
        for image_path in sorted(image_dir.iterdir()):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            if label_path.exists():
                pairs.append((image_path, label_path))
    if not split_found:
        image_dir = yolo_root / "images"
        label_dir = yolo_root / "labels"
        if image_dir.exists():
            for image_path in sorted(image_dir.iterdir()):
                if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    continue
                label_path = label_dir / f"{image_path.stem}.txt"
                if label_path.exists():
                    pairs.append((image_path, label_path))
    return pairs


def extract_crops(
    yolo_root: Path = YOLO_DIR,
    crops_dir: Path = CROPS_DIR,
    max_per_class: int = MAX_CROPS_PER_CLASS,
) -> dict[str, int]:
    yaml_path = find_data_yaml(yolo_root)
    with yaml_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    names = _parse_names(data.get("names", []))
    if crops_dir.exists():
        shutil.rmtree(crops_dir)
    crops_dir.mkdir(parents=True, exist_ok=True)

    collected: dict[str, list[dict[str, Any]]] = {name: [] for name in names}
    for image_path, label_path in _iter_yolo_pairs(yaml_path.parent):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(float(parts[0]))
            if class_id < 0 or class_id >= len(names):
                continue
            class_name = names[class_id]
            if class_name in SKIP_CLASSES:
                continue
            xc, yc, bw, bh = (float(value) for value in parts[1:5])
            box_w = bw * width
            box_h = bh * height
            if box_w < MIN_BOX_PX or box_h < MIN_BOX_PX:
                continue
            x1 = max(0, int((xc - bw / 2) * width) - 2)
            y1 = max(0, int((yc - bh / 2) * height) - 2)
            x2 = min(width, int((xc + bw / 2) * width) + 2)
            y2 = min(height, int((yc + bh / 2) * height) + 2)
            if x2 - x1 < MIN_BOX_PX or y2 - y1 < MIN_BOX_PX:
                continue
            collected.setdefault(class_name, []).append(
                {
                    "image_path": image_path,
                    "box": (x1, y1, x2, y2),
                    "area": (x2 - x1) * (y2 - y1),
                    "source_width": width,
                    "source_height": height,
                }
            )

    manifest: dict[str, Any] = {
        "source": DATASET_URL,
        "version": ROBOFLOW_VERSION,
        "classes": {},
    }
    counts: dict[str, int] = {}
    for class_name, samples in collected.items():
        if class_name in SKIP_CLASSES or not samples:
            continue
        chosen = _pick_diverse(samples, max_per_class)
        slug = class_slug(class_name)
        class_dir = crops_dir / slug
        class_dir.mkdir(parents=True, exist_ok=True)
        entries = []
        for index, sample in enumerate(chosen, start=1):
            image = cv2.imread(str(sample["image_path"]))
            if image is None:
                continue
            x1, y1, x2, y2 = sample["box"]
            crop = image[y1:y2, x1:x2]
            filename = f"{slug}_{index:02d}.png"
            output = class_dir / filename
            if not cv2.imwrite(str(output), crop):
                continue
            entries.append(
                {
                    "file": f"{slug}/{filename}",
                    "source_width": sample["source_width"],
                    "source_height": sample["source_height"],
                }
            )
        if entries:
            manifest["classes"][class_name] = entries
            counts[class_name] = len(entries)

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return counts


def _pick_diverse(samples: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if len(samples) <= limit:
        return samples
    median_area = sorted(sample["area"] for sample in samples)[len(samples) // 2]
    by_image: dict[Path, dict[str, Any]] = {}
    for sample in samples:
        current = by_image.get(sample["image_path"])
        if current is None or abs(sample["area"] - median_area) < abs(current["area"] - median_area):
            by_image[sample["image_path"]] = sample
    ranked = sorted(by_image.values(), key=lambda sample: abs(sample["area"] - median_area))
    return ranked[:limit]


def load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        return {"classes": {}}
    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    data.setdefault("classes", {})
    return data


def crop_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for source in (load_manifest(), load_custom_manifest()):
        for name, entries in source.get("classes", {}).items():
            counts[name] = counts.get(name, 0) + len(entries)
    return counts


def list_class_crop_paths(class_name: str) -> list[Path]:
    paths: list[Path] = []
    for manifest, root in (
        (load_manifest(), CROPS_DIR),
        (load_custom_manifest(), CUSTOM_DIR),
    ):
        for entry in manifest.get("classes", {}).get(class_name, []):
            path = root / entry["file"]
            if path.exists():
                paths.append(path)
    return paths


def list_gather_classes() -> list[str]:
    names = list(GATHER_CLASSES)
    for source in (load_manifest(), load_custom_manifest()):
        for name in source.get("classes", {}):
            if name not in names and name not in SKIP_CLASSES:
                names.append(name)
    return names


def is_custom_item(class_name: str) -> bool:
    return class_name in load_custom_manifest().get("classes", {})


def iter_selected_crops(enabled_items: list[str] | None) -> list[dict[str, Any]]:
    custom_manifest = load_custom_manifest()
    dataset_manifest = load_manifest()
    custom_classes = set(custom_manifest.get("classes", {}))
    results: list[dict[str, Any]] = []

    def append_from(manifest: dict[str, Any], root: Path, custom: bool, skip: set[str] | None = None) -> None:
        classes = manifest.get("classes", {})
        selected = set(enabled_items) if enabled_items is not None else set(classes)
        for class_name, entries in classes.items():
            if class_name not in selected:
                continue
            if skip and class_name in skip:
                continue
            for entry in entries:
                path = root / entry["file"]
                if not path.exists():
                    continue
                results.append(
                    {
                        "name": class_name,
                        "path": path,
                        "source_width": int(entry.get("source_width", 1920)),
                        "source_height": int(entry.get("source_height", 1080)),
                        "custom": custom,
                    }
                )

    # User photos override Roboflow crops for the same item.
    append_from(dataset_manifest, CROPS_DIR, custom=False, skip=custom_classes)
    append_from(custom_manifest, CUSTOM_DIR, custom=True)
    return results


def load_custom_manifest() -> dict[str, Any]:
    if not CUSTOM_MANIFEST_PATH.exists():
        return {"classes": {}}
    with CUSTOM_MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    data.setdefault("classes", {})
    return data


def save_custom_manifest(data: dict[str, Any]) -> None:
    CUSTOM_DIR.mkdir(parents=True, exist_ok=True)
    with CUSTOM_MANIFEST_PATH.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def add_custom_images(
    class_name: str,
    image_paths: list[Path],
    source_width: int,
    source_height: int,
) -> int:
    name = class_name.strip().lower()
    if not name:
        raise ValueError("Item name cannot be empty.")
    if name in SKIP_CLASSES:
        raise ValueError("That class is reserved.")
    slug = class_slug(name)
    class_dir = CUSTOM_DIR / slug
    class_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_custom_manifest()
    entries = list(manifest.setdefault("classes", {}).get(name, []))
    added = 0
    next_index = len(entries) + 1
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        filename = f"{slug}_{next_index:02d}.png"
        output = class_dir / filename
        if not cv2.imwrite(str(output), image):
            continue
        entries.append(
            {
                "file": f"{slug}/{filename}",
                "source_width": int(source_width),
                "source_height": int(source_height),
            }
        )
        next_index += 1
        added += 1
    if added == 0:
        raise ValueError("Could not read any of the selected images.")
    manifest["classes"][name] = entries
    save_custom_manifest(manifest)
    return added

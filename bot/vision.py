from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from bot.config import Settings, list_template_files
from bot.dataset import iter_selected_crops


@dataclass(frozen=True)
class Match:
    name: str
    confidence: float
    x: int
    y: int
    width: int
    height: int

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)


class TemplateMatcher:
    def __init__(self) -> None:
        self._templates: list[tuple[str, np.ndarray]] = []
        self._signature: tuple[object, ...] | None = None

    def reload_if_needed(self, settings: Settings, folder: Path, warnings: list[str]) -> None:
        scale = settings.template_scale()
        custom_files = [
            path
            for path in list_template_files(folder)
            if settings.enabled_templates is None or path.name in settings.enabled_templates
        ]
        dataset_crops = iter_selected_crops(settings.enabled_items)
        signature = (
            tuple((path.name, path.stat().st_mtime_ns, path.stat().st_size) for path in custom_files),
            tuple(
                (
                    str(item["path"]),
                    item["path"].stat().st_mtime_ns,
                    item["source_width"],
                    item["source_height"],
                    settings.game_width,
                    settings.game_height,
                )
                for item in dataset_crops
            ),
            scale,
            tuple(settings.enabled_items or []),
        )
        if signature == self._signature:
            return

        self._templates = []
        scale_x, scale_y = scale
        for path in custom_files:
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                warnings.append(f"Could not load template {path.name}")
                continue
            scaled = _scale_gray(image, scale_x, scale_y)
            self._templates.append((path.stem, scaled))

        for item in dataset_crops:
            image = cv2.imread(str(item["path"]), cv2.IMREAD_GRAYSCALE)
            if image is None:
                warnings.append(f"Could not load crop {item['path'].name}")
                continue
            item_scale_x = settings.game_width / max(1, int(item["source_width"]))
            item_scale_y = settings.game_height / max(1, int(item["source_height"]))
            scaled = _scale_gray(image, item_scale_x, item_scale_y)
            self._templates.append((str(item["name"]), scaled))

        self._signature = signature

    def find_best(
        self,
        frame_bgr: np.ndarray,
        confidence: float,
        warnings: list[str],
    ) -> Match | None:
        if not self._templates:
            return None
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        frame_h, frame_w = gray.shape[:2]
        best: Match | None = None
        for name, template in self._templates:
            th, tw = template.shape[:2]
            if th > frame_h or tw > frame_w:
                warnings.append(
                    f"Skipped {name}: template {tw}x{th} is larger than capture {frame_w}x{frame_h}"
                )
                continue
            result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(result)
            if max_val < confidence:
                continue
            if best is None or max_val > best.confidence:
                best = Match(
                    name=name,
                    confidence=float(max_val),
                    x=int(max_loc[0]),
                    y=int(max_loc[1]),
                    width=int(tw),
                    height=int(th),
                )
        return best


def _scale_gray(image: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    height, width = image.shape[:2]
    new_w = max(1, int(round(width * scale_x)))
    new_h = max(1, int(round(height * scale_y)))
    if (new_w, new_h) == (width, height):
        return image
    interpolation = cv2.INTER_AREA if (scale_x < 1 or scale_y < 1) else cv2.INTER_LINEAR
    return cv2.resize(image, (new_w, new_h), interpolation=interpolation)


def draw_debug(frame_bgr: np.ndarray, match: Match | None) -> np.ndarray:
    vis = frame_bgr.copy()
    if match is None:
        cv2.putText(vis, "no match", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return vis
    cv2.rectangle(
        vis,
        (match.x, match.y),
        (match.x + match.width, match.y + match.height),
        (0, 255, 0),
        2,
    )
    cx, cy = match.center
    cv2.drawMarker(vis, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 12, 2)
    label = f"{match.name} {match.confidence:.2f}"
    cv2.putText(vis, label, (match.x, max(20, match.y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return vis

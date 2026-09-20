from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from bot.config import Settings
from bot.dataset import iter_selected_crops

SCALES = (0.5, 0.7, 0.9, 1.15, 1.45)
SEARCH_SCALE = 0.55
MIN_TEMPLATE = 16
MAX_FRAME_FRACTION = 0.35
COLOR_GATE = 0.22
UI_LEFT = 0.06
UI_RIGHT = 0.94
UI_TOP = 0.07
UI_BOTTOM = 0.82


@dataclass(frozen=True)
class Match:
    name: str
    confidence: float
    x: int
    y: int
    width: int
    height: int
    color: float = 1.0

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)


@dataclass
class Probe:
    name: str
    confidence: float
    color: float
    x: int
    y: int
    width: int
    height: int
    accepted: bool


@dataclass
class _Template:
    name: str
    bgr: np.ndarray
    gray: np.ndarray
    base_scale: float


class TemplateMatcher:
    def __init__(self) -> None:
        self._templates: list[_Template] = []
        self._signature: tuple[object, ...] | None = None
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.last_probe: Probe | None = None

    def reload_if_needed(self, settings: Settings, warnings: list[str]) -> None:
        crops = iter_selected_crops(settings.enabled_items)
        signature = (
            tuple(
                (
                    str(item["path"]),
                    item["path"].stat().st_mtime_ns,
                    item["source_width"],
                    item["source_height"],
                    item.get("custom", False),
                    settings.game_width,
                    settings.game_height,
                )
                for item in crops
            ),
            tuple(settings.enabled_items or []),
        )
        if signature == self._signature:
            return

        self._templates = []
        for item in crops:
            image = cv2.imread(str(item["path"]), cv2.IMREAD_COLOR)
            if image is None:
                warnings.append(f"Could not load crop {item['path'].name}")
                continue
            if item.get("custom"):
                base_scale = 1.0
            else:
                base_scale = min(
                    settings.game_width / max(1, int(item["source_width"])),
                    settings.game_height / max(1, int(item["source_height"])),
                )
            gray = self._prepare_gray(image)
            self._templates.append(
                _Template(name=str(item["name"]), bgr=image, gray=gray, base_scale=base_scale)
            )
        self._signature = signature

    def find_best(
        self,
        frame_bgr: np.ndarray,
        confidence: float,
        warnings: list[str],
    ) -> Match | None:
        self.last_probe = None
        if not self._templates:
            return None
        frame_h, frame_w = frame_bgr.shape[:2]
        x0, y0, x1, y1 = _playable_roi(frame_w, frame_h)
        roi_bgr = frame_bgr[y0:y1, x0:x1]
        roi_gray = self._prepare_gray(roi_bgr)
        roi_h, roi_w = roi_gray.shape[:2]
        search_w = max(32, int(roi_w * SEARCH_SCALE))
        search_h = max(32, int(roi_h * SEARCH_SCALE))
        search_gray = cv2.resize(roi_gray, (search_w, search_h), interpolation=cv2.INTER_AREA)
        scale_x = roi_w / search_w
        scale_y = roi_h / search_h

        best_probe: Probe | None = None
        best_match: Match | None = None
        best_rank = -1.0

        for template in self._templates:
            for rel_scale in SCALES:
                scale = template.base_scale * rel_scale
                th, tw = template.gray.shape[:2]
                new_w = max(1, int(round(tw * scale)))
                new_h = max(1, int(round(th * scale)))
                if new_w < MIN_TEMPLATE or new_h < MIN_TEMPLATE:
                    continue
                if new_w > roi_w * MAX_FRAME_FRACTION or new_h > roi_h * MAX_FRAME_FRACTION:
                    continue
                if new_w >= roi_w or new_h >= roi_h:
                    continue
                gray_t = _resize_gray(template.gray, new_w, new_h)
                bgr_t = cv2.resize(template.bgr, (new_w, new_h), interpolation=_interp(scale))
                tw_s = max(8, int(round(new_w / scale_x)))
                th_s = max(8, int(round(new_h / scale_y)))
                if tw_s >= search_w or th_s >= search_h:
                    continue
                search_t = _resize_gray(gray_t, tw_s, th_s)
                result = cv2.matchTemplate(search_gray, search_t, cv2.TM_CCOEFF_NORMED)
                for shape_score, loc in _top_peaks(result, count=2, min_dist=max(8, min(tw_s, th_s) // 3)):
                    px = x0 + int(round(loc[0] * scale_x))
                    py = y0 + int(round(loc[1] * scale_y))
                    px = min(max(0, px), frame_w - new_w)
                    py = min(max(0, py), frame_h - new_h)
                    patch = frame_bgr[py : py + new_h, px : px + new_w]
                    if patch.shape[0] != new_h or patch.shape[1] != new_w:
                        continue
                    color_score = _color_score(patch, bgr_t)
                    rank = float(shape_score) * (0.55 + 0.45 * color_score)
                    probe = Probe(
                        name=template.name,
                        confidence=float(shape_score),
                        color=float(color_score),
                        x=px,
                        y=py,
                        width=new_w,
                        height=new_h,
                        accepted=False,
                    )
                    passes_color = color_score >= COLOR_GATE or _is_low_saturation(bgr_t)
                    if best_probe is None or rank > best_rank:
                        best_probe = probe
                        best_rank = rank
                    if shape_score >= confidence and passes_color:
                        if best_match is None or shape_score > best_match.confidence:
                            best_match = Match(
                                name=template.name,
                                confidence=float(shape_score),
                                x=px,
                                y=py,
                                width=new_w,
                                height=new_h,
                                color=float(color_score),
                            )

        if best_match is not None:
            self.last_probe = Probe(
                name=best_match.name,
                confidence=best_match.confidence,
                color=best_match.color,
                x=best_match.x,
                y=best_match.y,
                width=best_match.width,
                height=best_match.height,
                accepted=True,
            )
            return best_match

        self.last_probe = best_probe
        if best_probe is not None:
            warnings.append(
                f"Closest {best_probe.name}: shape {best_probe.confidence:.2f} "
                f"color {best_probe.color:.2f} (need shape {confidence:.2f})"
            )
        return None

    def _prepare_gray(self, bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = self._clahe.apply(gray)
        return cv2.GaussianBlur(gray, (3, 3), 0)


def _playable_roi(width: int, height: int) -> tuple[int, int, int, int]:
    x0 = int(width * UI_LEFT)
    y0 = int(height * UI_TOP)
    x1 = int(width * UI_RIGHT)
    y1 = int(height * UI_BOTTOM)
    return x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)


def _interp(scale: float) -> int:
    return cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR


def _resize_gray(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if image.shape[1] == width and image.shape[0] == height:
        return image
    return cv2.resize(image, (width, height), interpolation=_interp(width / max(1, image.shape[1])))


def _top_peaks(
    result: np.ndarray,
    count: int,
    min_dist: int,
) -> list[tuple[float, tuple[int, int]]]:
    peaks: list[tuple[float, tuple[int, int]]] = []
    work = result.copy()
    for _ in range(count):
        _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(work)
        if max_val < 0.25:
            break
        peaks.append((float(max_val), (int(max_loc[0]), int(max_loc[1]))))
        y, x = int(max_loc[1]), int(max_loc[0])
        y0 = max(0, y - min_dist)
        x0 = max(0, x - min_dist)
        work[y0 : y + min_dist, x0 : x + min_dist] = 0
    return peaks


def _is_low_saturation(bgr: np.ndarray) -> bool:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 1])) < 28


def _color_score(patch_bgr: np.ndarray, template_bgr: np.ndarray) -> float:
    if patch_bgr.size == 0 or template_bgr.size == 0:
        return 0.0
    if _is_low_saturation(template_bgr):
        patch_gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
        tmpl_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
        hist_p = cv2.calcHist([patch_gray], [0], None, [32], [0, 256])
        hist_t = cv2.calcHist([tmpl_gray], [0], None, [32], [0, 256])
    else:
        patch_hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
        tmpl_hsv = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2HSV)
        hist_p = cv2.calcHist([patch_hsv], [0, 1], None, [18, 16], [0, 180, 0, 256])
        hist_t = cv2.calcHist([tmpl_hsv], [0, 1], None, [18, 16], [0, 180, 0, 256])
    cv2.normalize(hist_p, hist_p)
    cv2.normalize(hist_t, hist_t)
    score = float(cv2.compareHist(hist_p, hist_t, cv2.HISTCMP_CORREL))
    return float(np.clip(score, 0.0, 1.0))


def draw_debug(frame_bgr: np.ndarray, match: Match | None, probe: Probe | None = None) -> np.ndarray:
    vis = frame_bgr.copy()
    x0, y0, x1, y1 = _playable_roi(vis.shape[1], vis.shape[0])
    cv2.rectangle(vis, (x0, y0), (x1, y1), (80, 80, 80), 1)
    target = match
    color = (0, 255, 0)
    if target is None and probe is not None:
        target = Match(
            name=probe.name,
            confidence=probe.confidence,
            x=probe.x,
            y=probe.y,
            width=probe.width,
            height=probe.height,
            color=probe.color,
        )
        color = (0, 165, 255)
    if target is None:
        cv2.putText(vis, "no match", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return vis
    cv2.rectangle(
        vis,
        (target.x, target.y),
        (target.x + target.width, target.y + target.height),
        color,
        2,
    )
    cx, cy = target.center
    cv2.drawMarker(vis, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 12, 2)
    label = f"{target.name} {target.confidence:.2f} col {target.color:.2f}"
    cv2.putText(vis, label, (target.x, max(20, target.y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return vis

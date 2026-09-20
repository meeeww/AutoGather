from __future__ import annotations

from typing import Any

import numpy as np
from mss import MSS

MonitorInfo = dict[str, Any]
Region = dict[str, int]
Sct = Any


def list_monitors() -> list[MonitorInfo]:
    with MSS() as sct:
        monitors = []
        for index, monitor in enumerate(sct.monitors):
            if index == 0:
                continue
            monitors.append(
                {
                    "index": index,
                    "left": int(monitor["left"]),
                    "top": int(monitor["top"]),
                    "width": int(monitor["width"]),
                    "height": int(monitor["height"]),
                }
            )
        return monitors


def monitor_label(info: MonitorInfo) -> str:
    return (
        f"Monitor {info['index']} "
        f"({info['width']}x{info['height']} at {info['left']},{info['top']})"
    )


def resolve_region(
    monitor: dict[str, int],
    game_width: int,
    game_height: int,
    anchor: str = "center",
) -> Region:
    width = max(1, min(int(game_width), int(monitor["width"])))
    height = max(1, min(int(game_height), int(monitor["height"])))
    if anchor == "top_left":
        left = int(monitor["left"])
        top = int(monitor["top"])
    else:
        left = int(monitor["left"]) + (int(monitor["width"]) - width) // 2
        top = int(monitor["top"]) + (int(monitor["height"]) - height) // 2
    return {"left": left, "top": top, "width": width, "height": height}


def grab_frame(
    sct: Sct,
    monitor_index: int,
    game_width: int,
    game_height: int,
    anchor: str = "center",
) -> tuple[np.ndarray, Region]:
    monitors = sct.monitors
    if monitor_index < 1 or monitor_index >= len(monitors):
        monitor_index = 1 if len(monitors) > 1 else 0
    region = resolve_region(monitors[monitor_index], game_width, game_height, anchor)
    shot = sct.grab(region)
    bgr = np.asarray(shot)[:, :, :3].copy()
    return bgr, region

from __future__ import annotations

import math
import random
import time

from pynput.mouse import Button, Controller

_mouse = Controller()


def click_screen(x: int, y: int, jitter: int = 3) -> tuple[int, int]:
    target_x = int(x) + random.randint(-jitter, jitter)
    target_y = int(y) + random.randint(-jitter, jitter)
    _mouse.position = (target_x, target_y)
    time.sleep(0.02)
    _mouse.click(Button.left, 1)
    return target_x, target_y


def wander_click(region: dict[str, int], step: int) -> tuple[int, int]:
    width = max(1, int(region["width"]))
    height = max(1, int(region["height"]))
    center_x = int(region["left"]) + width // 2
    center_y = int(region["top"]) + int(height * 0.52)
    angle = (step % 8) * (math.pi / 4) + random.uniform(-0.28, 0.28)
    shortest = min(width, height)
    radius = random.uniform(shortest * 0.18, shortest * 0.32)
    x = int(center_x + math.cos(angle) * radius)
    y = int(center_y + math.sin(angle) * radius)
    margin_x = int(width * 0.12)
    margin_top = int(height * 0.12)
    margin_bottom = int(height * 0.18)
    min_x = int(region["left"]) + margin_x
    max_x = int(region["left"]) + width - margin_x
    min_y = int(region["top"]) + margin_top
    max_y = int(region["top"]) + height - margin_bottom
    x = min(max(x, min_x), max(min_x, max_x))
    y = min(max(y, min_y), max(min_y, max_y))
    return click_screen(x, y, jitter=8)

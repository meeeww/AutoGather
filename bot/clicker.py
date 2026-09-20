from __future__ import annotations

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

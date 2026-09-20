from __future__ import annotations

import threading
import time
from collections.abc import Callable

import cv2
from mss import MSS

from bot.capture import grab_frame
from bot.clicker import WANDER_RECAST, click_screen, wander_click_forward
from bot.config import Settings
from bot.vision import TemplateMatcher, draw_debug

LogFn = Callable[[str], None]
StatusFn = Callable[[str], None]
SettingsFn = Callable[[], Settings]
StoppedFn = Callable[[], None]
DEBUG_WINDOW = "AutoGather debug"


class BotLoop:
    def __init__(
        self,
        get_settings: SettingsFn,
        log: LogFn,
        status: StatusFn,
        on_stopped: StoppedFn | None = None,
    ) -> None:
        self._get_settings = get_settings
        self._log = log
        self._status = status
        self._on_stopped = on_stopped
        self._running = threading.Event()
        self._wakeup = threading.Event()
        self._thread: threading.Thread | None = None
        self._matcher = TemplateMatcher()

    @property
    def running(self) -> bool:
        return self._running.is_set()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._running.set()
        self._wakeup.clear()
        self._thread = threading.Thread(target=self._run, name="autogather-loop", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        self._wakeup.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5.0)
        self._thread = None
        self._status("Stopped")

    def _sleep(self, seconds: float) -> None:
        deadline = time.monotonic() + max(0.0, seconds)
        while self._running.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            if self._wakeup.wait(timeout=min(0.05, remaining)):
                self._wakeup.clear()
                if not self._running.is_set():
                    break

    def _run(self) -> None:
        self._status("Running")
        self._log("Bot started")
        last_skip_log = 0.0
        debug_open = False
        wander_until = 0.0
        last_wander_click = 0.0
        zigzag_step = 0
        try:
            with MSS() as sct:
                while self._running.is_set():
                    settings = self._get_settings()
                    warnings: list[str] = []
                    self._matcher.reload_if_needed(settings, warnings)
                    for warning in warnings:
                        now = time.monotonic()
                        if now - last_skip_log >= 2.0:
                            self._log(warning)
                            last_skip_log = now

                    frame, region = grab_frame(
                        sct,
                        settings.monitor,
                        settings.game_width,
                        settings.game_height,
                        settings.capture_anchor,
                    )
                    match_warnings: list[str] = []
                    match = self._matcher.find_best(frame, settings.confidence, match_warnings)
                    if not self._running.is_set():
                        break
                    for warning in match_warnings:
                        now = time.monotonic()
                        if now - last_skip_log >= 2.0:
                            self._log(warning)
                            last_skip_log = now

                    if not self._running.is_set():
                        break
                    if settings.show_debug:
                        if not debug_open:
                            cv2.namedWindow(DEBUG_WINDOW, cv2.WINDOW_NORMAL)
                            debug_open = True
                        cv2.imshow(DEBUG_WINDOW, draw_debug(frame, match, self._matcher.last_probe))
                        cv2.waitKey(1)
                    elif debug_open:
                        self._close_debug()
                        debug_open = False

                    if match is None:
                        now = time.monotonic()
                        if settings.wander_enabled:
                            if wander_until <= now:
                                wander_until = now + settings.wander_duration
                                zigzag_step = 0
                                clicked_x, clicked_y = wander_click_forward(region, zigzag_step)
                                zigzag_step += 1
                                last_wander_click = now
                                self._status("Wandering")
                                self._log(
                                    f"No materials found, walking forward for {settings.wander_duration:.0f}s "
                                    f"-> click ({clicked_x}, {clicked_y})"
                                )
                            elif now - last_wander_click >= WANDER_RECAST:
                                clicked_x, clicked_y = wander_click_forward(region, zigzag_step)
                                zigzag_step += 1
                                last_wander_click = now
                                self._status("Wandering")
                            else:
                                self._status("Wandering")
                            self._sleep(settings.scan_interval)
                            continue
                        self._status("Scanning")
                        self._sleep(settings.scan_interval)
                        continue

                    wander_until = 0.0
                    last_wander_click = 0.0

                    # Click the lower part of the node so the character walks to it.
                    rel_x = match.x + match.width // 2
                    rel_y = match.y + int(match.height * 0.68)
                    screen_x = region["left"] + rel_x
                    screen_y = region["top"] + rel_y
                    clicked_x, clicked_y = click_screen(screen_x, screen_y)
                    self._status("Farming")
                    self._log(
                        f"Farming {match.name} ({match.confidence:.2f}) at ({clicked_x}, {clicked_y}) "
                        f"for {settings.wait_between_clicks:.0f}s"
                    )
                    self._sleep(settings.wait_between_clicks)
        except Exception as exc:
            self._log(f"Bot error: {exc}")
        finally:
            self._close_debug()
            if self._running.is_set():
                self._running.clear()
            self._status("Stopped")
            self._log("Bot stopped")
            if self._on_stopped is not None:
                self._on_stopped()

    @staticmethod
    def _close_debug() -> None:
        try:
            cv2.destroyWindow(DEBUG_WINDOW)
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

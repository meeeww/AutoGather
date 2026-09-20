from __future__ import annotations

import ctypes
import sys


def enable_dpi_awareness() -> None:
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except OSError:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except OSError:
            pass


def main() -> None:
    enable_dpi_awareness()
    from bot.gui import run_app

    run_app()


if __name__ == "__main__":
    main()

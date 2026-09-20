from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "config.json"
SECRETS_PATH = ROOT_DIR / "secrets.json"
TEMPLATES_DIR = ROOT_DIR / "templates"

TEMPLATE_BASE_WIDTH = 1920
TEMPLATE_BASE_HEIGHT = 1080
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


@dataclass
class Settings:
    monitor: int = 1
    game_width: int = 1920
    game_height: int = 1080
    capture_anchor: str = "center"
    wait_between_clicks: float = 1.5
    scan_interval: float = 0.4
    confidence: float = 0.82
    show_debug: bool = False
    enabled_templates: list[str] | None = field(default=None)
    enabled_items: list[str] | None = field(default=None)

    def template_scale(self) -> tuple[float, float]:
        return (
            self.game_width / TEMPLATE_BASE_WIDTH,
            self.game_height / TEMPLATE_BASE_HEIGHT,
        )


def default_settings() -> Settings:
    return Settings()


def load_settings(path: Path = CONFIG_PATH) -> Settings:
    if not path.exists():
        settings = default_settings()
        save_settings(settings, path)
        return settings
    with path.open("r", encoding="utf-8") as handle:
        data: dict[str, Any] = json.load(handle)
    defaults = asdict(default_settings())
    defaults.update({key: value for key, value in data.items() if key in defaults})
    return Settings(**defaults)


def save_settings(settings: Settings, path: Path = CONFIG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(settings), handle, indent=2)
        handle.write("\n")


def list_template_files(folder: Path = TEMPLATES_DIR) -> list[Path]:
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)
        return []
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )

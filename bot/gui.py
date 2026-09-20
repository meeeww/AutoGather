from __future__ import annotations

import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2
from mss import MSS
from pynput import keyboard

from bot.capture import grab_frame, list_monitors, monitor_label
from bot.config import CAPTURES_DIR, Settings, load_settings, save_settings
from bot.dataset import (
    DATASET_URL,
    add_custom_images,
    crop_counts,
    display_name,
    download_roboflow,
    extract_crops,
    import_folder,
    import_zip,
    list_gather_classes,
    list_class_crop_paths,
    load_api_key,
    save_api_key,
)
from bot.loop import BotLoop


class SettingsStore:
    def __init__(self, settings: Settings) -> None:
        self._lock = threading.Lock()
        self._settings = settings

    def set(self, settings: Settings) -> None:
        with self._lock:
            self._settings = settings

    def get(self) -> Settings:
        with self._lock:
            return self._settings


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("AutoGather")
        self.root.minsize(520, 820)
        self.root.attributes("-topmost", True)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.settings = load_settings()
        self.store = SettingsStore(self.settings)
        self.bot = BotLoop(
            self.store.get,
            self.log,
            self.set_status,
            on_stopped=lambda: self.root.after(0, self._on_bot_stopped),
        )
        self._item_vars: dict[str, tk.BooleanVar] = {}
        self._item_photos: list[tk.PhotoImage] = []
        self._known_items: set[str] = set(list_gather_classes())
        self._hotkey_listener: keyboard.Listener | None = None

        self._build()
        self._load_into_form(self.settings)
        self.refresh_items()
        self._bind_live_settings()
        self._start_hotkey()
        self._sync_store()
        self.set_status("Stopped")
        self.log("Ready. Import the dataset or add your own gather items, then press Start or F8.")

    def _build(self) -> None:
        pad = {"padx": 10, "pady": 4}
        main = ttk.Frame(self.root, padding=8)
        main.pack(fill="both", expand=True)

        capture = ttk.LabelFrame(main, text="Capture", padding=8)
        capture.pack(fill="x", **pad)

        ttk.Label(capture, text="Monitor").grid(row=0, column=0, sticky="w", pady=2)
        self.monitor_var = tk.StringVar()
        self.monitor_combo = ttk.Combobox(
            capture, textvariable=self.monitor_var, state="readonly", width=38
        )
        self.monitor_combo.grid(row=0, column=1, columnspan=3, sticky="ew", pady=2)

        ttk.Label(capture, text="Game resolution").grid(row=1, column=0, sticky="w", pady=2)
        self.width_var = tk.StringVar()
        self.height_var = tk.StringVar()
        ttk.Entry(capture, textvariable=self.width_var, width=8).grid(row=1, column=1, sticky="w", pady=2)
        ttk.Label(capture, text="x").grid(row=1, column=2, sticky="w", padx=4)
        ttk.Entry(capture, textvariable=self.height_var, width=8).grid(row=1, column=3, sticky="w", pady=2)

        ttk.Label(capture, text="Region position").grid(row=2, column=0, sticky="w", pady=2)
        self.anchor_var = tk.StringVar()
        self.anchor_combo = ttk.Combobox(
            capture,
            textvariable=self.anchor_var,
            state="readonly",
            values=("center", "top_left"),
            width=16,
        )
        self.anchor_combo.grid(row=2, column=1, columnspan=3, sticky="w", pady=2)
        capture.columnconfigure(1, weight=1)

        timing = ttk.LabelFrame(main, text="Timing", padding=8)
        timing.pack(fill="x", **pad)

        ttk.Label(timing, text="Wait between clicks (s)").grid(row=0, column=0, sticky="w", pady=2)
        self.wait_var = tk.StringVar()
        ttk.Entry(timing, textvariable=self.wait_var, width=10).grid(row=0, column=1, sticky="w", pady=2)

        ttk.Label(timing, text="Scan interval (s)").grid(row=1, column=0, sticky="w", pady=2)
        self.scan_var = tk.StringVar()
        ttk.Entry(timing, textvariable=self.scan_var, width=10).grid(row=1, column=1, sticky="w", pady=2)

        ttk.Label(timing, text="Match confidence").grid(row=2, column=0, sticky="w", pady=2)
        self.confidence_var = tk.DoubleVar(value=0.82)
        self.confidence_label = ttk.Label(timing, text="0.82")
        slider = ttk.Scale(
            timing,
            from_=0.50,
            to=0.99,
            variable=self.confidence_var,
            command=self._on_confidence,
        )
        slider.grid(row=2, column=1, sticky="ew", pady=2)
        self.confidence_label.grid(row=2, column=2, sticky="w", padx=6)
        timing.columnconfigure(1, weight=1)

        ttk.Label(timing, text="Wander duration (s)").grid(row=3, column=0, sticky="w", pady=2)
        self.wander_wait_var = tk.StringVar()
        ttk.Entry(timing, textvariable=self.wander_wait_var, width=10).grid(row=3, column=1, sticky="w", pady=2)

        self.debug_var = tk.BooleanVar(value=False)
        self.wander_var = tk.BooleanVar(value=True)
        flags = ttk.Frame(main)
        flags.pack(fill="x", **pad)
        ttk.Checkbutton(flags, text="Show debug window", variable=self.debug_var).pack(side="left")
        ttk.Checkbutton(
            flags,
            text="Wander when nothing is found",
            variable=self.wander_var,
        ).pack(side="left", padx=(16, 0))

        items = ttk.LabelFrame(main, text="Items to gather", padding=8)
        items.pack(fill="both", expand=True, **pad)
        self.items_status = ttk.Label(items, text="Dataset not imported yet.")
        self.items_status.pack(anchor="w")
        item_btns = ttk.Frame(items)
        item_btns.pack(fill="x", pady=(6, 4))
        ttk.Button(item_btns, text="Import dataset", command=self.open_import_dialog).pack(side="left")
        ttk.Button(item_btns, text="Add item", command=self.open_add_item_dialog).pack(side="left", padx=6)
        ttk.Button(item_btns, text="All", command=lambda: self._set_all_items(True)).pack(side="left")
        ttk.Button(item_btns, text="None", command=lambda: self._set_all_items(False)).pack(side="left", padx=6)

        items_holder = ttk.Frame(items)
        items_holder.pack(fill="both", expand=True)
        self.items_canvas = tk.Canvas(items_holder, height=280, highlightthickness=0)
        items_scroll = ttk.Scrollbar(items_holder, orient="vertical", command=self.items_canvas.yview)
        self.items_inner = ttk.Frame(self.items_canvas)
        self._items_window = self.items_canvas.create_window((0, 0), window=self.items_inner, anchor="nw")
        self.items_inner.bind(
            "<Configure>",
            lambda _event: self.items_canvas.configure(scrollregion=self.items_canvas.bbox("all")),
        )
        self.items_canvas.bind(
            "<Configure>",
            lambda event: self.items_canvas.itemconfigure(self._items_window, width=event.width),
        )
        self.items_canvas.configure(yscrollcommand=items_scroll.set)
        self.items_canvas.pack(side="left", fill="both", expand=True)
        items_scroll.pack(side="right", fill="y")
        self._bind_mousewheel(self.items_canvas)

        buttons = ttk.Frame(main)
        buttons.pack(fill="x", **pad)
        self.start_btn = ttk.Button(buttons, text="Start", command=self.start_bot)
        self.start_btn.pack(side="left", padx=(0, 6))
        self.stop_btn = ttk.Button(buttons, text="Stop", command=self.stop_bot, state="disabled")
        self.stop_btn.pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Save screenshot of region", command=self.save_screenshot).pack(side="left")

        status_row = ttk.Frame(main)
        status_row.pack(fill="x", **pad)
        self.status_var = tk.StringVar(value="Stopped")
        ttk.Label(status_row, text="Status:").pack(side="left")
        ttk.Label(status_row, textvariable=self.status_var).pack(side="left", padx=6)
        ttk.Label(status_row, text="Hotkey: F8").pack(side="right")

        log_frame = ttk.LabelFrame(main, text="Log", padding=6)
        log_frame.pack(fill="both", expand=False, **pad)
        self.log_text = tk.Text(log_frame, height=7, state="disabled", wrap="word")
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scroll.pack(side="right", fill="y")

        self._populate_monitors()

    def _populate_monitors(self) -> None:
        self._monitors = list_monitors()
        labels = [monitor_label(info) for info in self._monitors]
        self.monitor_combo["values"] = labels
        if not labels:
            self.monitor_var.set("")
            return
        index = self.settings.monitor
        chosen = next((info for info in self._monitors if info["index"] == index), self._monitors[0])
        self.monitor_var.set(monitor_label(chosen))

    def _selected_monitor_index(self) -> int:
        selected = self.monitor_var.get()
        for info in self._monitors:
            if monitor_label(info) == selected:
                return int(info["index"])
        return 1

    def _on_confidence(self, _value: str) -> None:
        self.confidence_label.configure(text=f"{float(self.confidence_var.get()):.2f}")

    def _load_into_form(self, settings: Settings) -> None:
        self.width_var.set(str(settings.game_width))
        self.height_var.set(str(settings.game_height))
        self.anchor_var.set(settings.capture_anchor if settings.capture_anchor in {"center", "top_left"} else "center")
        self.wait_var.set(str(settings.wait_between_clicks))
        self.scan_var.set(str(settings.scan_interval))
        self.confidence_var.set(settings.confidence)
        self.confidence_label.configure(text=f"{settings.confidence:.2f}")
        self.debug_var.set(settings.show_debug)
        self.wander_var.set(settings.wander_enabled)
        self.wander_wait_var.set(str(settings.wander_duration))

    def refresh_items(self) -> None:
        for child in self.items_inner.winfo_children():
            child.destroy()
        self._item_photos = []
        previous = {name: var.get() for name, var in self._item_vars.items()}
        saved = self.settings.enabled_items
        self._item_vars = {}
        counts = crop_counts()
        classes = list_gather_classes()
        for class_name in classes:
            enabled = previous.get(class_name)
            if enabled is None:
                if saved is None or class_name in saved or class_name not in self._known_items:
                    enabled = True
                else:
                    enabled = False
            var = tk.BooleanVar(value=enabled)
            self._item_vars[class_name] = var
            count = counts.get(class_name, 0)
            row = ttk.Frame(self.items_inner)
            row.pack(fill="x", pady=(0, 8), padx=2)
            header = ttk.Frame(row)
            header.pack(fill="x")
            ttk.Checkbutton(
                header,
                text=f"{display_name(class_name)} ({count})",
                variable=var,
                command=self._sync_store,
            ).pack(side="left")
            ttk.Button(
                header,
                text="Add photos",
                command=lambda n=class_name: self.open_add_item_dialog(n),
            ).pack(side="left", padx=8)
            thumbs = ttk.Frame(row)
            thumbs.pack(anchor="w", pady=(4, 0))
            paths = list_class_crop_paths(class_name)
            if not paths:
                ttk.Label(thumbs, text="No images yet. Import the dataset or add photos.").pack(anchor="w")
                continue
            for path in paths:
                photo = self._thumb_photo(path, 64)
                if photo is None:
                    continue
                self._item_photos.append(photo)
                thumb = tk.Label(thumbs, image=photo, cursor="hand2", bd=1, relief="solid")
                thumb.pack(side="left", padx=(0, 6))
                thumb.bind("<Button-1>", lambda _event, p=path, n=class_name: self._show_crop_preview(n, p))
        total = sum(counts.values())
        if total:
            self.items_status.configure(
                text=f"AlbionGathering v4 ready: {total} crops. Click a thumbnail to enlarge."
            )
        else:
            self.items_status.configure(
                text="Import the dataset or add your own item photos to get started."
            )
        self.items_inner.update_idletasks()
        self.items_canvas.configure(scrollregion=self.items_canvas.bbox("all"))
        self._known_items = set(classes)

    def _thumb_photo(self, path: Path, size: int) -> tk.PhotoImage | None:
        image = cv2.imread(str(path))
        if image is None:
            return None
        height, width = image.shape[:2]
        scale = size / max(height, width, 1)
        new_w = max(1, int(round(width * scale)))
        new_h = max(1, int(round(height * scale)))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        y0 = (size - new_h) // 2
        x0 = (size - new_w) // 2
        canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
        ok, buffer = cv2.imencode(".png", canvas)
        if not ok:
            return None
        return tk.PhotoImage(data=buffer.tobytes())

    def _show_crop_preview(self, class_name: str, path: Path) -> None:
        image = cv2.imread(str(path))
        if image is None:
            messagebox.showerror("Preview failed", f"Could not read {path.name}", parent=self.root)
            return
        height, width = image.shape[:2]
        max_side = 360
        scale = min(1.0, max_side / max(height, width, 1))
        if scale != 1.0:
            image = cv2.resize(
                image,
                (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        ok, buffer = cv2.imencode(".png", image)
        if not ok:
            return
        photo = tk.PhotoImage(data=buffer.tobytes())
        win = tk.Toplevel(self.root)
        win.title(f"{display_name(class_name)} — {path.name}")
        win.attributes("-topmost", True)
        win.resizable(False, False)
        label = ttk.Label(win, image=photo)
        label.image = photo
        label.pack(padx=10, pady=10)
        ttk.Label(win, text=str(path)).pack(padx=10, pady=(0, 10))

    def _bind_mousewheel(self, canvas: tk.Canvas) -> None:
        def on_wheel(event: tk.Event) -> None:
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind("<Enter>", lambda _event: canvas.bind_all("<MouseWheel>", on_wheel))
        canvas.bind("<Leave>", lambda _event: canvas.unbind_all("<MouseWheel>"))

    def _set_all_items(self, enabled: bool) -> None:
        for var in self._item_vars.values():
            var.set(enabled)
        self._sync_store()

    def enabled_item_names(self) -> list[str]:
        return [name for name, var in self._item_vars.items() if var.get()]

    def read_form(self) -> Settings:
        try:
            game_width = int(self.width_var.get().strip())
            game_height = int(self.height_var.get().strip())
            wait_between = float(self.wait_var.get().strip())
            scan_interval = float(self.scan_var.get().strip())
            wander_duration = float(self.wander_wait_var.get().strip())
        except ValueError as exc:
            raise ValueError("Resolution, wait time, and scan interval must be numbers.") from exc
        if game_width < 1 or game_height < 1:
            raise ValueError("Game resolution must be at least 1x1.")
        if wait_between <= 0 or scan_interval <= 0 or wander_duration <= 0:
            raise ValueError("Wait, scan, and wander duration must be greater than 0.")
        confidence = float(self.confidence_var.get())
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Confidence must be between 0 and 1.")
        anchor = self.anchor_var.get().strip() or "center"
        if anchor not in {"center", "top_left"}:
            anchor = "center"
        return Settings(
            monitor=self._selected_monitor_index(),
            game_width=game_width,
            game_height=game_height,
            capture_anchor=anchor,
            wait_between_clicks=wait_between,
            scan_interval=scan_interval,
            confidence=confidence,
            show_debug=bool(self.debug_var.get()),
            enabled_items=self.enabled_item_names(),
            wander_enabled=bool(self.wander_var.get()),
            wander_duration=wander_duration,
        )

    def apply_settings(self, persist: bool = True) -> Settings:
        settings = self.read_form()
        self.settings = settings
        self.store.set(settings)
        if persist:
            save_settings(settings)
        return settings

    def _bind_live_settings(self) -> None:
        for variable in (
            self.monitor_var,
            self.width_var,
            self.height_var,
            self.anchor_var,
            self.wait_var,
            self.scan_var,
            self.confidence_var,
            self.debug_var,
            self.wander_var,
            self.wander_wait_var,
        ):
            variable.trace_add("write", lambda *_args: self._sync_store())

    def _sync_store(self) -> None:
        try:
            self.store.set(self.read_form())
        except ValueError:
            return

    def start_bot(self) -> None:
        if self.bot.running:
            return
        try:
            settings = self.apply_settings(persist=True)
        except ValueError as exc:
            messagebox.showerror("Invalid settings", str(exc), parent=self.root)
            return
        enabled_items = settings.enabled_items or []
        dataset_ready = sum(crop_counts().get(name, 0) for name in enabled_items)
        if dataset_ready == 0:
            if enabled_items:
                messagebox.showerror(
                    "No images",
                    "Selected items have no photos yet. Import the dataset or add photos.",
                    parent=self.root,
                )
            else:
                messagebox.showerror(
                    "Nothing selected",
                    "Select at least one gather item.",
                    parent=self.root,
                )
            return
        self.bot.start()
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")

    def stop_bot(self) -> None:
        try:
            if self.bot.running:
                self.bot.stop()
        except Exception as exc:
            self.log(f"Stop failed: {exc}")
        try:
            self.start_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")
            self.status_var.set("Stopped")
        except tk.TclError:
            return

    def toggle_bot(self) -> None:
        if self.bot.running:
            self.stop_bot()
        else:
            self.start_bot()

    def open_add_item_dialog(self, item_name: str = "") -> None:
        win = tk.Toplevel(self.root)
        win.title("Add gather item")
        win.attributes("-topmost", True)
        win.resizable(False, False)
        frame = ttk.Frame(win, padding=12)
        frame.pack(fill="both", expand=True)
        ttk.Label(
            frame,
            wraplength=400,
            text="Give the item a name, then upload cropped photos of it. Use tight crops of the node, not the whole screen.",
        ).pack(anchor="w")
        ttk.Label(frame, text="Item name").pack(anchor="w", pady=(10, 2))
        name_var = tk.StringVar(value=item_name)
        name_entry = ttk.Entry(frame, textvariable=name_var, width=40)
        name_entry.pack(anchor="w")
        if item_name:
            name_entry.configure(state="readonly")

        def add_files() -> None:
            paths = filedialog.askopenfilenames(
                parent=win,
                title="Choose item photos",
                filetypes=[
                    ("Images", "*.png *.jpg *.jpeg *.bmp *.webp"),
                    ("All files", "*.*"),
                ],
            )
            if not paths:
                return
            name = name_var.get().strip()
            try:
                settings = self.read_form()
                added = add_custom_images(
                    name,
                    [Path(path) for path in paths],
                    settings.game_width,
                    settings.game_height,
                )
            except ValueError as exc:
                messagebox.showerror("Could not add photos", str(exc), parent=win)
                return
            key = name.strip().lower()
            self.refresh_items()
            if key in self._item_vars:
                self._item_vars[key].set(True)
            self._sync_store()
            self.log(f"Added {added} photo(s) to {display_name(key)}")
            win.destroy()

        ttk.Button(frame, text="Choose photos...", command=add_files).pack(anchor="w", pady=(12, 0))

    def open_import_dialog(self) -> None:
        win = tk.Toplevel(self.root)
        win.title("Import AlbionGathering dataset")
        win.attributes("-topmost", True)
        win.resizable(False, False)
        frame = ttk.Frame(win, padding=12)
        frame.pack(fill="both", expand=True)
        ttk.Label(
            frame,
            wraplength=420,
            text=(
                "This uses Roboflow AlbionGathering v4 (CC BY 4.0): granite, iron ore, "
                "rough log, sandstone, tin ore, titanium, travertine, and resources. "
                "Monsters are skipped. Download needs a free Roboflow API key, or point "
                "at a YOLO zip/folder you already exported."
            ),
        ).pack(anchor="w")
        ttk.Label(frame, text=DATASET_URL, foreground="#3366aa").pack(anchor="w", pady=(4, 8))

        ttk.Label(frame, text="Roboflow API key").pack(anchor="w")
        key_var = tk.StringVar(value=load_api_key())
        ttk.Entry(frame, textvariable=key_var, width=52, show="*").pack(anchor="w", pady=(0, 8))

        status = ttk.Label(frame, text="")
        buttons = ttk.Frame(frame)
        buttons.pack(fill="x", pady=(0, 6))

        def set_busy(busy: bool, message: str = "") -> None:
            state = "disabled" if busy else "normal"
            for child in buttons.winfo_children():
                child.configure(state=state)
            status.configure(text=message)

        def run_job(job, start_message: str) -> None:
            set_busy(True, start_message)

            def work() -> None:
                try:
                    counts = job()
                except Exception as exc:
                    self.root.after(
                        0,
                        lambda err=exc: (
                            set_busy(False, ""),
                            messagebox.showerror("Import failed", str(err), parent=win),
                        ),
                    )
                    return
                self.root.after(0, lambda: self._finish_import(win, counts, set_busy))

            threading.Thread(target=work, daemon=True).start()

        def download() -> None:
            api_key = key_var.get().strip()
            if not api_key:
                messagebox.showerror(
                    "API key required",
                    "Create a free key at https://app.roboflow.com/settings/api",
                    parent=win,
                )
                return
            save_api_key(api_key)

            def job():
                download_roboflow(api_key)
                return extract_crops()

            run_job(job, "Downloading AlbionGathering v4...")

        def browse_zip() -> None:
            path = filedialog.askopenfilename(
                parent=win,
                title="Select YOLO dataset zip",
                filetypes=[("Zip files", "*.zip"), ("All files", "*.*")],
            )
            if not path:
                return

            def job():
                import_zip(Path(path))
                return extract_crops()

            run_job(job, "Extracting zip...")

        def browse_folder() -> None:
            path = filedialog.askdirectory(parent=win, title="Select YOLO dataset folder")
            if not path:
                return

            def job():
                import_folder(Path(path))
                return extract_crops()

            run_job(job, "Importing folder...")

        ttk.Button(buttons, text="Download v4", command=download).pack(side="left")
        ttk.Button(buttons, text="Load zip", command=browse_zip).pack(side="left", padx=6)
        ttk.Button(buttons, text="Load folder", command=browse_folder).pack(side="left")
        status.pack(anchor="w")

    def _finish_import(self, win: tk.Toplevel, counts: dict[str, int], set_busy) -> None:
        try:
            if win.winfo_exists():
                set_busy(False, "")
                win.destroy()
        except tk.TclError:
            pass
        self.refresh_items()
        self._sync_store()
        total = sum(counts.values())
        self.log(f"Imported dataset: {total} crops across {len(counts)} classes")

    def save_screenshot(self) -> None:
        try:
            settings = self.apply_settings(persist=True)
        except ValueError as exc:
            messagebox.showerror("Invalid settings", str(exc), parent=self.root)
            return
        CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = CAPTURES_DIR / f"capture_{stamp}.png"
        with MSS() as sct:
            frame, region = grab_frame(
                sct,
                settings.monitor,
                settings.game_width,
                settings.game_height,
                settings.capture_anchor,
            )
        if not cv2.imwrite(str(output), frame):
            messagebox.showerror("Save failed", f"Could not write {output}", parent=self.root)
            return
        self.log(f"Saved {output} — crop it, then Add item / Add photos")

    def log(self, message: str) -> None:
        def append() -> None:
            try:
                if not self.root.winfo_exists():
                    return
                stamp = time.strftime("%H:%M:%S")
                self.log_text.configure(state="normal")
                self.log_text.insert("end", f"{stamp}  {message}\n")
                self.log_text.see("end")
                self.log_text.configure(state="disabled")
            except tk.TclError:
                return

        try:
            self.root.after(0, append)
        except tk.TclError:
            return

    def set_status(self, text: str) -> None:
        def apply() -> None:
            try:
                if self.root.winfo_exists():
                    self.status_var.set(text)
            except tk.TclError:
                return

        try:
            self.root.after(0, apply)
        except tk.TclError:
            return

    def _on_bot_stopped(self) -> None:
        try:
            if not self.root.winfo_exists():
                return
            self.start_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")
            self.status_var.set("Stopped")
        except tk.TclError:
            return

    def _start_hotkey(self) -> None:
        def on_press(key: keyboard.Key | keyboard.KeyCode | None) -> None:
            if key == keyboard.Key.f8:
                self.root.after(0, self.toggle_bot)

        self._hotkey_listener = keyboard.Listener(on_press=on_press)
        self._hotkey_listener.daemon = True
        self._hotkey_listener.start()

    def on_close(self) -> None:
        try:
            settings = self.read_form()
            save_settings(settings)
        except ValueError:
            save_settings(self.settings)
        self.stop_bot()
        listener = self._hotkey_listener
        if listener is not None:
            listener.stop()
        self.root.destroy()


def run_app() -> None:
    root = tk.Tk()
    try:
        root.call("tk", "scaling", 1.0)
    except tk.TclError:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    run_app()

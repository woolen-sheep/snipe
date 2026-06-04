from __future__ import annotations

import hashlib
import threading
import shutil
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse
from typing import Any, Final

import webview
from loguru import logger

from .config import config
from .image_io import collect_images, load_image
from .metadata import extract_regions_from_xmp
from .paths import CACHE_DIR, WEB_PREVIEW_DIR


IS_WINDOWS = sys.platform == "win32"

# Win32 constants used by the preview icon patch. pywebview's WinForms backend
# does not expose named equivalents for these message and class-icon values.
PREVIEW_APP_USER_MODEL_ID: Final[str] = "com.snipe.preview"
WM_SETICON: Final[int] = 0x0080
ICON_SMALL: Final[int] = 0
ICON_BIG: Final[int] = 1
GCLP_HICON: Final[int] = -14
GCLP_HICONSM: Final[int] = -34


def _install_winforms_window_icon(icon_path: Path) -> None:
    if not IS_WINDOWS or not icon_path.exists():
        return

    import ctypes

    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(PREVIEW_APP_USER_MODEL_ID)
    except Exception as exc:
        logger.debug("Failed to set preview AppUserModelID {}: {}", PREVIEW_APP_USER_MODEL_ID, exc)

    try:
        import webview.platforms.winforms as winforms
    except Exception as exc:
        logger.debug("Preview window icon patch unavailable: {}", exc)
        return

    try:
        system_drawing = __import__("System.Drawing", fromlist=["Bitmap", "Icon"])
        Bitmap = system_drawing.Bitmap
        Icon = system_drawing.Icon
    except Exception as exc:
        logger.debug("System.Drawing is unavailable for preview icon patch: {}", exc)
        return

    browser_form = winforms.BrowserView.BrowserForm
    if getattr(browser_form, "_snipe_icon_patch", False):
        return

    original_init = browser_form.__init__

    def patched_init(self, window, cache_dir):
        original_init(self, window, cache_dir)

        bitmap = None
        icon_handle = None
        try:
            bitmap = Bitmap(str(icon_path))
            icon_handle = bitmap.GetHicon()
            self._snipe_window_icon = Icon.FromHandle(icon_handle).Clone()
            self.Icon = self._snipe_window_icon
            handle = int(self.Handle.ToInt64())
            hicon = int(self._snipe_window_icon.Handle.ToInt64())
            ctypes.windll.user32.SendMessageW(handle, WM_SETICON, ICON_SMALL, hicon)
            ctypes.windll.user32.SendMessageW(handle, WM_SETICON, ICON_BIG, hicon)
            ctypes.windll.user32.SetClassLongPtrW(handle, GCLP_HICONSM, hicon)
            ctypes.windll.user32.SetClassLongPtrW(handle, GCLP_HICON, hicon)
        except Exception as exc:
            logger.debug("Failed to apply preview window icon {}: {}", icon_path, exc)
        finally:
            if bitmap is not None:
                bitmap.Dispose()
            if icon_handle is not None:
                ctypes.windll.user32.DestroyIcon(int(icon_handle.ToInt64()))

    browser_form.__init__ = patched_init
    browser_form._snipe_icon_patch = True


def _resolve_preview_path(url_path: str) -> Path:
    relative_path = Path(*Path(unquote(urlparse(url_path).path).lstrip("/")).parts)
    if not relative_path.parts:
        relative_path = Path("index.html")

    if relative_path.parts[0] == "cache":
        return (CACHE_DIR / Path(*relative_path.parts[1:])).resolve()

    return (WEB_PREVIEW_DIR / relative_path).resolve()


def _is_path_within(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root.resolve())
        return True
    except ValueError:
        return False


class PreviewHTTPRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_PREVIEW_DIR), **kwargs)

    def translate_path(self, path: str) -> str:
        resolved = _resolve_preview_path(path)

        if _is_path_within(CACHE_DIR, resolved) or _is_path_within(WEB_PREVIEW_DIR, resolved):
            return str(resolved)

        return str((WEB_PREVIEW_DIR / "__missing__").resolve())

    def log_message(self, format: str, *args) -> None:
        logger.debug("Preview HTTP {}", format % args)


def start_preview_http_server() -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", 0), PreviewHTTPRequestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True, name="snipe-preview-http")
    thread.start()
    logger.info("Preview HTTP server listening at http://127.0.0.1:{}", server.server_port)
    return server


def make_image_uri(path: Path) -> tuple[str, tuple[int, int]]:
    img = load_image(path)
    size = img.size

    stat = path.stat()
    digest = hashlib.md5(f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8")).hexdigest()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{digest}.jpg"

    if not cache_file.exists():
        converted = img.convert("RGB")
        converted.save(cache_file, format="JPEG", quality=90)
        converted.close()
        logger.debug("Cached preview image {} -> {}", path, cache_file)

    img.close()
    return f"/cache/{cache_file.name}", size


class PreviewAPI:
    def __init__(self, images: list[Path]):
        self.images = images
        self.index = 0
        self._initial_payload = self._payload()

    def _payload(self) -> dict[str, Any]:
        path = self.images[self.index]
        uri, size = make_image_uri(path)
        regions = extract_regions_from_xmp(path, size)
        payload = {
            "fileName": path.name,
            "path": str(path),
            "index": self.index,
            "total": len(self.images),
            "imageUri": uri,
            "imageWidth": size[0],
            "imageHeight": size[1],
            "regions": regions,
        }
        logger.debug(
            "Preview payload for {} ({} / {}): size={}x{}, regions={}, uri_prefix={}...",
            path,
            self.index + 1,
            len(self.images),
            size[0],
            size[1],
            len(regions),
            uri[:32],
        )
        return payload

    def current(self) -> dict[str, Any]:
        if self._initial_payload is not None:
            payload = self._initial_payload
            self._initial_payload = None
            return payload

        return self._payload()

    def next(self) -> dict[str, Any]:
        self.index = (self.index + 1) % len(self.images)
        return self._payload()

    def previous(self) -> dict[str, Any]:
        self.index = (self.index - 1) % len(self.images)
        return self._payload()


def launch_preview(images: list[Path], debug: bool = False) -> None:
    html_path = WEB_PREVIEW_DIR / "index.html"
    window_icon = WEB_PREVIEW_DIR / "snipe.ico"

    if not html_path.exists():
        logger.error("Preview frontend not found at {}", html_path)
        sys.exit(1)

    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR, ignore_errors=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    _install_winforms_window_icon(window_icon)

    api = PreviewAPI(images)
    server = start_preview_http_server()
    preview_url = f"http://127.0.0.1:{server.server_port}/index.html"
    logger.info("Launching preview for {} images (debug={})", len(images), debug)
    webview.create_window("Snipe Preview", preview_url, js_api=api, width=1280, height=900, resizable=True)
    try:
        webview.start(debug=debug, icon=str(window_icon) if window_icon.exists() else None)
    finally:
        server.shutdown()
        server.server_close()


def run_preview() -> None:
    preview_config = config.preview

    if preview_config.source is None:
        logger.error("Source path is not configured")
        sys.exit(1)

    if not preview_config.source.exists():
        logger.error("Source path does not exist: {}", preview_config.source)
        sys.exit(1)

    images = collect_images(preview_config.source)
    if not images:
        logger.error("No images found in {}", preview_config.source)
        sys.exit(1)

    launch_preview(images, debug=preview_config.debug)

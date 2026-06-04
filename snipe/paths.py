from __future__ import annotations

import os
import sys
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
RESOURCE_DIR = PACKAGE_DIR / "resources"
WEB_DIR = PACKAGE_DIR / "web"
DATA_DIR = RESOURCE_DIR
WEB_PREVIEW_DIR = WEB_DIR / "preview"


def _resolve_user_config_dir() -> Path:
	if sys.platform == "win32":
		appdata = os.environ.get("APPDATA")
		if appdata:
			return Path(appdata)
		return Path.home() / "AppData" / "Roaming"

	xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
	if xdg_config_home:
		return Path(xdg_config_home)

	return Path.home() / ".config"


CONFIG_DIR = _resolve_user_config_dir() / ".snipe"
CACHE_DIR = CONFIG_DIR / "cache"
MODELS_DIR = CONFIG_DIR / "models"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

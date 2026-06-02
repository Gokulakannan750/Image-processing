"""
config/config_manager.py
========================
Loads and validates YAML configuration.
Provides a global config dict to the rest of the application.

Hot-reload: call config_manager.start_watch() to start a background thread that
reloads the YAML whenever its mtime changes.  Works on all platforms (no SIGHUP).
"""

import os
import threading
import time
import yaml
from typing import Any, Dict, Optional
from utils.logger import get_logger

log = get_logger(__name__)


def _validate(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Validate raw config dict against the pydantic schema.  Returns the
    validated dict (with defaults filled in).  Logs warnings on error but
    never raises so the app can still start with partial config."""
    try:
        from config.schema import AppConfig

        validated = AppConfig(**raw)
        # Return as plain dict so the rest of the app is unchanged
        return validated.model_dump()
    except ImportError:
        log.warning("pydantic not installed — skipping config schema validation.")
        return raw
    except Exception as exc:  # catches ValidationError and any other parse error
        log.warning(
            "Config validation warnings (defaults used for invalid fields):\n%s", exc
        )
        return raw


class ConfigManager:
    """Manages application configuration from YAML files."""

    def __init__(self, config_path: Optional[str] = None):
        self._config: Dict[str, Any] = {}
        self._reload_callbacks: list = (
            []
        )  # callables invoked after every successful reload
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "default.yaml")

        self.config_path = config_path
        self.load()

    def on_reload(self, callback) -> None:
        """Register a zero-argument callable that is invoked after every config reload."""
        self._reload_callbacks.append(callback)

    def load(self) -> None:
        """Loads or reloads the configuration from the YAML file."""
        if not os.path.exists(self.config_path):
            log.error(f"Configuration file not found: {self.config_path}")
            self._config = {}
            return

        try:
            with open(self.config_path, "r") as f:
                raw = yaml.safe_load(f) or {}
            self._config = _validate(raw)
            log.info(f"Loaded and validated configuration from {self.config_path}")
            self._fire_reload_callbacks()
        except Exception as e:
            log.error(f"Failed to parse YAML config {self.config_path}: {e}")
            self._config = {}

    def _fire_reload_callbacks(self) -> None:
        for cb in self._reload_callbacks:
            try:
                cb()
            except Exception as exc:
                log.warning("Config reload callback %s raised: %s", cb, exc)

    # ── Hot-reload ────────────────────────────────────────────────────────────

    def start_watch(self, interval_s: float = 2.0) -> None:
        """Start a background thread that reloads config when the YAML file changes.

        Safe to call multiple times — only one watcher runs at a time.
        Works on Windows, macOS, and Linux (no SIGHUP needed).
        """
        if getattr(self, "_watcher_running", False):
            return
        self._watcher_running = True
        self._watch_interval = interval_s
        t = threading.Thread(
            target=self._watch_loop, name="config-watcher", daemon=True
        )
        t.start()
        log.info(
            "Config file watcher started (poll every %.1fs): %s",
            interval_s,
            self.config_path,
        )

    def _watch_loop(self) -> None:
        last_mtime: Optional[float] = self._file_mtime()
        while self._watcher_running:
            time.sleep(self._watch_interval)
            current = self._file_mtime()
            if current is not None and current != last_mtime:
                log.info("Config file changed — reloading.")
                self.load()
                last_mtime = current

    def _file_mtime(self) -> Optional[float]:
        try:
            return os.path.getmtime(self.config_path)
        except OSError:
            return None

    def stop_watch(self) -> None:
        self._watcher_running = False

    @property
    def config(self) -> Dict[str, Any]:
        """Access the loaded configuration dictionary."""
        return self._config

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Safely retrieve nested config values using dot notation.
        Example: get("detectors.aruco.marker_length_m", 0.20)
        """
        keys = key_path.split(".")
        val = self._config
        for key in keys:
            if isinstance(val, dict) and key in val:
                val = val[key]
            else:
                log.debug(
                    f"Config key '{key_path}' not found, using default: {default}"
                )
                return default
        return val


# Global instance
config_manager = ConfigManager()

"""
config/config_manager.py
========================
Loads and validates YAML configuration.
Provides a global config dict to the rest of the application.
"""
import os
import yaml
from typing import Any, Dict
from utils.logger import get_logger

log = get_logger(__name__)

class ConfigManager:
    """Manages application configuration from YAML files."""
    
    def __init__(self, config_path: str = None):
        self._config: Dict[str, Any] = {}
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "default.yaml")
        
        self.config_path = config_path
        self.load()

    def load(self) -> None:
        """Loads or reloads the configuration from the YAML file."""
        if not os.path.exists(self.config_path):
            log.error(f"Configuration file not found: {self.config_path}")
            # Provide an empty dict to prevent catastrophic failure, though many things will fail downstream
            self._config = {}
            return

        try:
            with open(self.config_path, "r") as f:
                self._config = yaml.safe_load(f) or {}
            log.info(f"Loaded configuration from {self.config_path}")
        except Exception as e:
            log.error(f"Failed to parse YAML config {self.config_path}: {e}")
            self._config = {}

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
                log.debug(f"Config key '{key_path}' not found, using default: {default}")
                return default
        return val

# Global instance
config_manager = ConfigManager()

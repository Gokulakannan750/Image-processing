from config.config_manager import ConfigManager
import os

def test_config_manager_defaults():
    # Use the default yaml we just created
    cm = ConfigManager()
    
    # Check default camera config
    assert cm.get("camera.index") == 0
    
    # Check non-existent key with default fallback
    assert cm.get("non.existent.key", "default_val") == "default_val"
    
def test_config_manager_invalid_path():
    # Provide a fake path, should not crash but return defaults
    cm = ConfigManager(config_path="fake_path.yaml")
    assert cm.config == {}
    assert cm.get("camera.index", 99) == 99

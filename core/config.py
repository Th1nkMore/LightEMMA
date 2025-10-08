"""
Configuration management for LightEMMA
"""
import os
import yaml
from typing import Dict, Any


class ConfigManager:
    """Manages configuration loading and validation"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Validate required sections
        required_sections = ['api_keys', 'model_path', 'data', 'prediction']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")

        return config

    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation (e.g., 'data.version')"""
        keys = key_path.split('.')
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def update(self, key_path: str, value: Any):
        """Update configuration value using dot notation"""
        keys = key_path.split('.')
        config = self.config

        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        # Set the value
        config[keys[-1]] = value

    def save(self, path: str = None):
        """Save current configuration to file"""
        save_path = path or self.config_path
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

    @property
    def api_keys(self):
        return self.config.get('api_keys', {})

    @property
    def model_path(self):
        return self.config.get('model_path', {})

    @property
    def data_config(self):
        return self.config.get('data', {})

    @property
    def prediction_config(self):
        return self.config.get('prediction', {})
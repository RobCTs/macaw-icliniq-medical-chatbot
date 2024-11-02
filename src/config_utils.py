# src/config_utils.py

import yaml
import json

def load_yaml_config(filepath):
    """Load a YAML configuration file."""
    with open(filepath, 'r') as file:
        return yaml.safe_load(file)

def load_json_config(filepath):
    """Load a JSON configuration file."""
    with open(filepath, 'r') as file:
        return json.load(file)

import yaml
from dotenv import load_dotenv
import os
from pathlib import Path

class Config():
    def __init__(self):
        load_dotenv()
        self.filepath = Path(os.getenv("config"))
        self.data = self._load_yaml()

    def _load_yaml(self):
        """Load YAML file contents and return as a Python dict."""
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {self.filepath}")
        
        with open(self.filepath, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)

    def __getitem__(self, key):
        """Allow dict-like access, e.g. loader['some_key']"""
        return self.data.get(key)

    def __repr__(self):
        return f"YamlLoader(filepath='{self.filepath}', keys={list(self.data.keys())})"
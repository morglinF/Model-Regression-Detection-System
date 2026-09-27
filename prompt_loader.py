"""YAML-backed PromptLoader: reads prompts/<prompt_id>/<version>.yaml."""

import re
from pathlib import Path
from typing import Optional

import yaml

from contracts import PromptConfig, PromptLoader

PROMPTS_DIR = Path(__file__).parent / "prompts"
VERSION_RE = re.compile(r"^v(\d+)$")


class YamlPromptLoader(PromptLoader):
    def __init__(self, prompts_dir: Path = PROMPTS_DIR):
        self.prompts_dir = Path(prompts_dir)

    def list_versions(self, prompt_id: str) -> list[str]:
        versions = [
            p.stem
            for p in (self.prompts_dir / prompt_id).glob("v*.yaml")
            if VERSION_RE.match(p.stem)
        ]
        return sorted(versions, key=lambda v: int(VERSION_RE.match(v).group(1)))

    def load(self, prompt_id: str, version: Optional[str] = None) -> PromptConfig:
        if version is None:
            versions = self.list_versions(prompt_id)
            if not versions:
                raise FileNotFoundError(f"No versions found in {self.prompts_dir / prompt_id}")
            version = versions[-1]

        path = self.prompts_dir / prompt_id / f"{version}.yaml"
        with open(path, encoding="utf-8") as f:
            config = PromptConfig.model_validate(yaml.safe_load(f))

        if config.id != prompt_id or config.version != version:
            raise ValueError(
                f"{path} declares id={config.id!r} version={config.version!r}, "
                f"which doesn't match its location"
            )
        if "{email}" not in config.user_template:
            raise ValueError(f"{path}: user_template must contain {{email}}")
        return config

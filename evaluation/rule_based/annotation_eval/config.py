from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


DEFAULT_CONFIG = {
    "display_name": "",
    "report_title": "",
    "dataset_code_dir": "dataset_code",
    "dataset_code_removed_dir": "dataset_code_removed",
    "dataset_image_removed_dir": "dataset_image_removed",
    "test_code_dir": "test_code",
    "test_image_dir": "test_code_image",
    "annotations_dir": "outputs/annotations",
    "gt_annotations_dir": "",
    "rendered_images_dir": "outputs/rendered_images",
    "analysis_dir": "outputs/analysis",
    "shared_text_extraction_dir": "",
    "execution_cwd": ".",
}


@lru_cache(maxsize=None)
def load_project_config(project_root: str | Path) -> dict[str, str]:
    root = Path(project_root).resolve()
    config = DEFAULT_CONFIG.copy()
    config_path = root / "pipeline_config.json"

    if config_path.exists():
        data = json.loads(config_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for key in DEFAULT_CONFIG:
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    config[key] = value.strip()

    if not config["display_name"]:
        config["display_name"] = root.name
    if not config["report_title"]:
        config["report_title"] = f"{config['display_name']} evaluation report"
    return config


def get_config_value(project_root: str | Path, key: str, default: str) -> str:
    return load_project_config(project_root).get(key, default) or default


def get_path(project_root: str | Path, key: str, default: str) -> Path:
    root = Path(project_root).resolve()
    return root / get_config_value(root, key, default)


def get_report_title(project_root: str | Path) -> str:
    return get_config_value(project_root, "report_title", "evaluation report")

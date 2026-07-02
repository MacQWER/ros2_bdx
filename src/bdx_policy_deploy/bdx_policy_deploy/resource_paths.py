from __future__ import annotations

from pathlib import Path

from ament_index_python.packages import get_package_share_directory


def resolve_resource_path(path_value: str | Path) -> Path:
    path_text = str(path_value)
    prefix = "package://"
    if path_text.startswith(prefix):
        package_and_path = path_text[len(prefix) :]
        package_name, separator, relative_path = package_and_path.partition("/")
        if not package_name or not separator or not relative_path:
            raise ValueError(f"Invalid package resource URI: {path_text}")
        return (Path(get_package_share_directory(package_name)) / relative_path).resolve()

    return Path(path_text).expanduser().resolve()

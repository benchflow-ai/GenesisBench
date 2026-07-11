from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class TaskDocumentError(ValueError):
    """Raised when a GenesisBench task.md document is invalid."""


@dataclass(frozen=True)
class TaskDocument:
    path: Path
    frontmatter: dict[str, Any]
    instruction: str

    @classmethod
    def from_path(cls, path: str | Path) -> TaskDocument:
        document_path = Path(path)
        frontmatter, instruction = _split_task_document(
            document_path.read_text()
        )
        return cls(
            path=document_path,
            frontmatter=frontmatter,
            instruction=instruction,
        )

    @property
    def package_name(self) -> str:
        task = _mapping(self.frontmatter.get("task"), "task")
        value = task.get("name")
        if not isinstance(value, str) or not value:
            raise TaskDocumentError("task.name must be a non-empty string")
        return value

    @property
    def genesisbench(self) -> dict[str, Any]:
        metadata = _mapping(self.frontmatter.get("metadata"), "metadata")
        return _mapping(metadata.get("genesisbench"), "metadata.genesisbench")


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TaskDocumentError(f"{field} must be a mapping")
    return value


def _split_task_document(text: str) -> tuple[dict[str, Any], str]:
    normalized = text.replace("\r\n", "\n")
    lines = normalized.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        raise TaskDocumentError("task.md must start with YAML frontmatter")

    closing_index = next(
        (
            index
            for index, line in enumerate(lines[1:], start=1)
            if line.strip() == "---"
        ),
        None,
    )
    if closing_index is None:
        raise TaskDocumentError("task.md frontmatter is missing closing ---")

    frontmatter_text = "".join(lines[1:closing_index])
    try:
        loaded = yaml.safe_load(frontmatter_text)
    except yaml.YAMLError as error:
        raise TaskDocumentError(
            f"task.md frontmatter is not valid YAML: {error}"
        ) from error
    if not isinstance(loaded, dict):
        raise TaskDocumentError("task.md frontmatter must be a mapping")

    body = "".join(lines[closing_index + 1 :]).lstrip("\n").strip()
    if not body:
        raise TaskDocumentError("task.md prompt body must not be empty")
    return loaded, body


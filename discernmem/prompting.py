from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from common.memory_guidance import (
    guidance_snapshot,
    mas_system_addendum,
    supported_mas_types,
)


PROMPT_DIR = Path(__file__).resolve().parent / "prompt"


def _read_prompt_file(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _markdown_section(text: str, *section_names: str) -> str:
    """Extract one level-two Markdown section from a prompt specification."""
    wanted = {name.strip().casefold() for name in section_names}
    lines = text.splitlines()
    start = None
    for index, line in enumerate(lines):
        if not line.startswith("## "):
            continue
        name = line[3:].strip().casefold()
        if name in wanted:
            start = index + 1
            break
    if start is None:
        raise ValueError("missing prompt section: " + ", ".join(sorted(section_names)))
    end = len(lines)
    for index in range(start, len(lines)):
        if lines[index].startswith("## "):
            end = index
            break
    return "\n".join(lines[start:end]).strip()


def _system_section(text: str) -> str:
    if "## System" not in text:
        return text.strip()
    return _markdown_section(text, "System", "System Prompt", "System Addendum")


def _output_contract_section(text: str) -> str:
    if "## Output Contract" not in text:
        return ""
    return _markdown_section(text, "Output Contract")


def load_prompt(
    filename: str,
    *,
    mas_type: str = "",
    include_mas_guidance: bool = False,
    enable_mas_guidance: bool = True,
    system_addenda: tuple[str, ...] = (),
) -> str:
    specification = _read_prompt_file(PROMPT_DIR / filename)
    parts = [_system_section(specification)]
    if include_mas_guidance and enable_mas_guidance:
        normalized = str(mas_type or "").strip().lower()
        if normalized not in supported_mas_types():
            raise ValueError(f"unsupported DiscernMem MAS guidance: {normalized!r}")
        parts.append(mas_system_addendum(normalized))
    parts.extend(str(value).strip() for value in system_addenda if str(value).strip())
    contract = _output_contract_section(specification)
    if contract:
        parts.append(f"### Output Contract\n{contract}")
    return "\n\n".join(parts)


def load_user_template(filename: str) -> str:
    text = _read_prompt_file(PROMPT_DIR / filename)
    return _markdown_section(text, "User", "User Prompt")


def render_user_prompt(
    filename: str,
    payload: Any,
    *,
    indent: int | None = 2,
    sort_keys: bool = True,
) -> str:
    """Render the canonical user section with one structured JSON payload."""
    template = load_user_template(filename)
    if template.count("{input_json}") != 1:
        raise ValueError(
            f"DiscernMem user template must contain one {{input_json}}: {filename}"
        )
    separators = (",", ":") if indent is None else None
    input_json = json.dumps(
        payload,
        ensure_ascii=False,
        indent=indent,
        sort_keys=sort_keys,
        separators=separators,
        default=str,
    )
    return template.replace("{input_json}", input_json)


def load_schema(filename: str) -> dict[str, Any]:
    payload = json.loads(
        (PROMPT_DIR / "schemas" / filename).read_text(encoding="utf-8")
    )
    if not isinstance(payload, dict):
        raise ValueError(f"DiscernMem schema must be a JSON object: {filename}")
    return payload


def prompt_bundle_fingerprint() -> str:
    """Hash every prompt, schema, and MAS/task guidance text used by the method."""
    digest = hashlib.sha256()
    paths = sorted(
        path
        for path in PROMPT_DIR.rglob("*")
        if path.is_file()
        and path.suffix in {".md", ".json"}
        and path.name != "README.md"
    )
    for path in paths:
        digest.update(path.relative_to(PROMPT_DIR).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    digest.update(b"guidance\0")
    digest.update(json.dumps(guidance_snapshot(), sort_keys=True).encode("utf-8"))
    return digest.hexdigest()

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .invariants import require_single_role
from .schema import MetaMemory


@dataclass(frozen=True)
class RequiredHandoff:
    from_role: str
    to_role: str
    purpose: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RequiredHandoff":
        return cls(
            from_role=str(data.get("from_role") or ""),
            to_role=str(data.get("to_role") or ""),
            purpose=str(data.get("purpose") or ""),
        )


@dataclass(frozen=True)
class ExtractionCandidate:
    memory: str
    roles: tuple[str, ...]
    use_when: tuple[str, ...]
    avoid_when: tuple[str, ...]
    evidence_refs: tuple[str, ...]

    @staticmethod
    def _strings(values: object) -> tuple[str, ...]:
        if not isinstance(values, (list, tuple)):
            return ()
        return tuple(
            dict.fromkeys(text for value in values if (text := str(value).strip()))
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExtractionCandidate":
        normalized_roles = cls._strings(data.get("roles", ()))
        role = require_single_role(normalized_roles, context="extraction candidate")
        return cls(
            memory=str(data.get("memory") or "").strip(),
            roles=(role,),
            use_when=cls._strings(data.get("use_when")),
            avoid_when=cls._strings(data.get("avoid_when")),
            evidence_refs=cls._strings(data.get("evidence_refs")),
        )

    @property
    def body(self) -> str:
        return self.memory

    @property
    def initial_meta(self) -> MetaMemory:
        return MetaMemory(
            roles=self.roles,
            use_when=self.use_when,
            avoid_when=self.avoid_when,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory": self.memory,
            "roles": list(self.roles),
            "use_when": list(self.use_when),
            "avoid_when": list(self.avoid_when),
            "evidence_refs": list(self.evidence_refs),
        }


def _require_nonblank_strings(
    candidate: Mapping[str, Any], field: str, *, allow_empty: bool
) -> None:
    values = candidate.get(field)
    if not isinstance(values, list):
        raise ValueError(f"DiscernMem extraction field {field!r} must be an array")
    if not allow_empty and not values:
        raise ValueError(f"DiscernMem extraction field {field!r} must not be empty")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(
            f"DiscernMem extraction field {field!r} contains an empty value"
        )


def candidate_from_response(
    payload: Mapping[str, Any], *, target_role: str = ""
) -> ExtractionCandidate | None:
    if not isinstance(payload, Mapping):
        raise ValueError("DiscernMem extraction must return one JSON object")
    decision = payload.get("decision")
    if decision not in {"STORE", "NO_MEMORY"}:
        raise ValueError(
            "DiscernMem extraction field decision must be STORE or NO_MEMORY"
        )
    memory = payload.get("memory")
    if not isinstance(memory, str) or not memory.strip():
        raise ValueError("DiscernMem extraction field memory must not be empty")
    if decision == "NO_MEMORY":
        if memory.strip() != "NO_MEMORY":
            raise ValueError("NO_MEMORY decision must use the NO_MEMORY sentinel")
        _require_nonblank_strings(payload, "use_when", allow_empty=True)
        _require_nonblank_strings(payload, "avoid_when", allow_empty=True)
        if payload.get("use_when") or payload.get("avoid_when"):
            raise ValueError("NO_MEMORY must have empty use_when and avoid_when")
        return None
    if memory.strip() == "NO_MEMORY":
        raise ValueError("STORE decision cannot use the NO_MEMORY sentinel")
    if not str(target_role).strip():
        raise ValueError("role-conditioned extraction requires target_role")
    _require_nonblank_strings(payload, "use_when", allow_empty=False)
    _require_nonblank_strings(payload, "avoid_when", allow_empty=True)
    return ExtractionCandidate.from_dict(
        {**dict(payload), "roles": [str(target_role).strip()]}
    )

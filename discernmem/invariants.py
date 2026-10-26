from __future__ import annotations

from enum import Enum
from typing import Iterable


class InvariantViolation(ValueError):
    """Raised when a DiscernMem method invariant would be violated."""


class MemoryState(str, Enum):
    PROVISIONAL = "provisional"
    ACTIVE = "active"
    INACTIVE = "inactive"


def require_single_role(
    roles: Iterable[str], *, context: str = "memory identity"
) -> str:
    normalized = tuple(
        dict.fromkeys(str(value).strip() for value in roles if str(value).strip())
    )
    if len(normalized) != 1:
        raise InvariantViolation(
            f"{context} must belong to exactly one role; got {list(normalized)!r}"
        )
    return normalized[0]


def retrieval_allowed(state: MemoryState | str, *, provisional_enabled: bool = True) -> bool:
    normalized = MemoryState(state)
    return normalized is MemoryState.ACTIVE or (
        provisional_enabled and normalized is MemoryState.PROVISIONAL
    )


def assert_main_effect_evidence(*, is_source_task: bool, h_label: str | None) -> None:
    if is_source_task:
        raise InvariantViolation("source tasks cannot contribute to main effect")
    if str(h_label or "").upper() == "H3":
        raise InvariantViolation("H3 boundary evidence cannot contribute to main effect")


def assert_linear_parent(*, current_version: int, expected_parent: int) -> None:
    if int(current_version) != int(expected_parent):
        raise InvariantViolation(
            f"stale version parent {expected_parent}; current version is {current_version}"
        )

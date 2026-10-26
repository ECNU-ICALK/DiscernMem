from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .ids import content_hash, event_id as make_event_id
from .storage import DiscernMemStore, read_json, safe_component, utc_now


class EventConflictError(RuntimeError):
    pass


@dataclass(frozen=True)
class LearningEvent:
    event_id: str
    event_type: str
    task_id: str
    input_hash: str
    status: str
    created_at: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LearningEvent":
        return cls(
            event_id=str(data["event_id"]),
            event_type=str(data["event_type"]),
            task_id=str(data["task_id"]),
            input_hash=str(data["input_hash"]),
            status=str(data["status"]),
            created_at=str(data["created_at"]),
        )


@dataclass(frozen=True)
class StageRecord:
    event_id: str
    stage: str
    input_hash: str
    output: Mapping[str, Any]
    completed_at: str


class EventJournal:
    def __init__(self, store: DiscernMemStore):
        self.store = store

    def _event_dir(self, event_id: str):
        return self.store.root / "events" / safe_component(event_id)

    def begin(
        self, event_type: str, task_id: str, inputs: Mapping[str, Any]
    ) -> LearningEvent:
        identifier = make_event_id(event_type, task_id, inputs)
        input_digest = content_hash(inputs)
        path = self._event_dir(identifier) / "event.json"
        with self.store._exclusive():
            if path.exists():
                existing = LearningEvent.from_dict(read_json(path))
                if (
                    existing.input_hash != input_digest
                    or existing.event_type != event_type
                ):
                    raise EventConflictError(
                        "event ID was reused with different semantic input"
                    )
                return existing
            payload = {
                "event_id": identifier,
                "event_type": event_type,
                "task_id": task_id,
                "input_hash": input_digest,
                "status": "in_progress",
                "created_at": utc_now(),
            }
            self.store._write_once(path, payload)
            return LearningEvent.from_dict(payload)

    def get_stage(self, event_id: str, stage: str) -> StageRecord | None:
        path = self._event_dir(event_id) / "stages" / f"{safe_component(stage)}.json"
        if not path.exists():
            return None
        data = read_json(path)
        return StageRecord(
            event_id=data["event_id"],
            stage=data["stage"],
            input_hash=data["input_hash"],
            output=dict(data.get("output") or {}),
            completed_at=data["completed_at"],
        )

    def run_stage(
        self,
        event: LearningEvent,
        stage: str,
        inputs: Mapping[str, Any],
        operation: Callable[[], Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        digest = content_hash(inputs)
        existing = self.get_stage(event.event_id, stage)
        if existing:
            if existing.input_hash != digest:
                raise EventConflictError(
                    f"event stage {stage!r} was retried with different input"
                )
            return existing.output
        output = dict(operation())
        path = (
            self._event_dir(event.event_id) / "stages" / f"{safe_component(stage)}.json"
        )
        payload = {
            "event_id": event.event_id,
            "stage": stage,
            "input_hash": digest,
            "output": output,
            "completed_at": utc_now(),
        }
        with self.store._exclusive():
            self.store._write_once(path, payload)
        return output

    def complete(self, event: LearningEvent, summary: Mapping[str, Any]) -> None:
        path = self._event_dir(event.event_id) / "completion.json"
        with self.store._exclusive():
            if path.exists():
                existing = read_json(path)
                if existing.get("status") != "completed" or dict(
                    existing.get("summary") or {}
                ) != dict(summary):
                    raise EventConflictError(
                        "event completion was retried with a different summary"
                    )
                return
            self.store._write_once(
                path,
                {
                    "event_id": event.event_id,
                    "status": "completed",
                    "summary": dict(summary),
                    "completed_at": utc_now(),
                },
            )

    def is_complete(self, event_id: str) -> bool:
        return (self._event_dir(event_id) / "completion.json").exists()

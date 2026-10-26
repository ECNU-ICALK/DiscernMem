from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

from .storage import DiscernMemStore, _atomic_write_json, read_json, utc_now


class DiscernMemMetrics:
    def __init__(self, store: DiscernMemStore):
        self.store = store

    @staticmethod
    def _json_files(path: Path):
        return sorted(path.rglob("*.json")) if path.exists() else []

    def compute(self) -> dict[str, Any]:
        identities = self.store.iter_identities()
        versions = [self.store.get_version(item.memory_id) for item in identities]
        state_counts = Counter(version.state.value for version in versions if version)

        admissions = [
            read_json(path) for path in self._json_files(self.store.root / "admissions")
        ]
        admission_counts = Counter(item.get("decision") for item in admissions)
        admission_quality_status = Counter(
            item.get("validation_status", "not_evaluated") for item in admissions
        )

        extraction_files = self._json_files(self.store.root / "audit" / "extractions")
        extractions = [read_json(path) for path in extraction_files]
        empty_candidates = sum(item.get("candidate") is None for item in extractions)

        replays = self.store.iter_replays()
        replay_conditions = Counter(item.condition.value for item in replays)
        replay_h_labels = Counter(item.h_label.value if item.h_label else "direct" for item in replays)
        replay_failures = len(self._json_files(self.store.root / "replays" / "attempts")) - len(replays)

        plans = [
            read_json(path) for path in self._json_files(self.store.root / "validation_plans")
        ]
        h_task_use = Counter(
            assignment["task_id"]
            for plan in plans
            for assignment in plan.get("assignments", [])
        )
        repeated_h_selections = sum(max(count - 1, 0) for count in h_task_use.values())

        cpe = [
            read_json(path) for path in self._json_files(self.store.root / "audit" / "cpe_selections")
        ]
        cpe_candidates = sum(len(item.get("evaluations", [])) for item in cpe)
        cpe_accepts = sum(bool(item.get("winner_candidate_id")) for item in cpe)
        cpe_stage_rejections = Counter(
            evaluation.get("failed_stage") or "accepted"
            for item in cpe
            for evaluation in item.get("evaluations", [])
        )
        direct_overwrites = len(
            self._json_files(self.store.root / "audit" / "direct_overwrite")
        )

        event_stages = [
            read_json(path)
            for path in self._json_files(self.store.root / "events")
            if path.parent.name == "stages"
        ]
        formal_outputs = [
            dict(item.get("output") or {})
            for item in event_stages
            if (item.get("output") or {}).get("routing")
        ]
        formal_routing = Counter(item.get("routing") for item in formal_outputs)
        repair_outcomes = Counter(
            item.get("status")
            for item in formal_outputs
            if str(item.get("status") or "").startswith("repair_")
            or item.get("status") == "direct_overwrite_committed"
        )

        retrievals = [
            read_json(path) for path in self._json_files(self.store.root / "audit" / "retrievals")
        ]
        provisional_uses = 0
        conflict_count = 0
        for retrieval in retrievals:
            state_by_id = {
                item["memory_id"]: item.get("state")
                for item in retrieval.get("candidate_pool", [])
            }
            provisional_uses += sum(
                state_by_id.get(item["memory_id"]) == "provisional"
                for item in retrieval.get("selected", [])
            )
            conflict_count += len(retrieval.get("conflicts", []))

        llm_audits = [
            read_json(path)
            for path in self._json_files(self.store.root / "audit")
            if "system_prompt" in read_json(path)
        ]
        prompt_tokens = sum(
            int((item.get("actual_usage") or {}).get("prompt_tokens") or 0)
            for item in llm_audits
        )
        completion_tokens = sum(
            int((item.get("actual_usage") or {}).get("completion_tokens") or 0)
            for item in llm_audits
        )
        llm_latency = sum(float(item.get("latency_seconds") or 0.0) for item in llm_audits)

        final_ids = set(self.store.config.split_manifest.final_test_task_ids)
        leaked_plan_tasks = sorted(
            final_ids
            & {
                assignment["task_id"]
                for plan in plans
                for assignment in plan.get("assignments", [])
            }
        )
        leaked_replay_tasks = sorted(final_ids & {item.task_id for item in replays})
        leaked_learning_tasks = sorted(
            final_ids
            & {path.stem for path in self._json_files(self.store.root / "tasks")}
        )
        leakage = {
            "validation_plan_task_ids": leaked_plan_tasks,
            "replay_task_ids": leaked_replay_tasks,
            "learning_task_ids": leaked_learning_tasks,
            "passed": not (leaked_plan_tasks or leaked_replay_tasks or leaked_learning_tasks),
        }

        state_transitions = Counter()
        for identity in identities:
            for version_dir in sorted(
                (self.store.root / "versions" / identity.memory_id).glob("v*")
            ):
                previous = None
                for evidence_path in sorted((version_dir / "evidence").glob("rev_*.json")):
                    state = read_json(evidence_path).get("state")
                    if previous is not None and state != previous:
                        state_transitions[f"{previous}->{state}"] += 1
                    previous = state

        return {
            "generated_at": utc_now(),
            "namespace": "discernmem",
            "config_fingerprint": self.store.config.fingerprint,
            "configuration": self.store.config.to_dict(),
            "memories": {
                "total": len(identities),
                "states": dict(state_counts),
                "formal_versions": sum(
                    len(list((self.store.root / "versions" / item.memory_id).glob("v*")))
                    for item in identities
                ),
            },
            "extraction": {
                "tasks": len(extractions),
                "empty_candidates": empty_candidates,
                "empty_candidate_rate": empty_candidates / len(extractions) if extractions else None,
                "quality_validation_mode": "prompt_only",
            },
            "admission": {
                "total": len(admissions),
                "decisions": dict(admission_counts),
                "quality_validation_mode": "prompt_only",
                "quality_validation_status": dict(admission_quality_status),
                "identity_accuracy": None,
                "identity_accuracy_reason": "requires an external labeled identity set",
            },
            "retrieval": {
                "calls": len(retrievals),
                "provisional_uses": provisional_uses,
                "conflict_groups": conflict_count,
            },
            "replay": {
                "successful_records": len(replays),
                "conditions": dict(replay_conditions),
                "h_labels": dict(replay_h_labels),
                "failed_attempt_files": max(replay_failures, 0),
            },
            "validation": {
                "plans": len(plans),
                "repeated_h_task_selections": repeated_h_selections,
            },
            "cpe": {
                "events": len(cpe),
                "candidates": cpe_candidates,
                "accepted_updates": cpe_accepts,
                "stage_outcomes": dict(cpe_stage_rejections),
                "current_counterexample_repair_rate": (
                    cpe_accepts / len(cpe) if cpe else None
                ),
                "h1_rejections": cpe_stage_rejections.get("H1", 0),
                "h2_rejections": cpe_stage_rejections.get("H2", 0),
                "direct_overwrite_updates": direct_overwrites,
            },
            "formal_use": {
                "assessed_events": len(formal_outputs),
                "routing": dict(formal_routing),
                "repair_outcomes": dict(repair_outcomes),
            },
            "state": {
                "transitions": dict(state_transitions),
            },
            "ablation": {
                "evolution_mode": self.store.config.evolution_mode,
                "enable_capability_constraints": self.store.config.enable_capability_constraints,
                "enable_h1_validation": self.store.config.enable_h1_validation,
                "enable_h2_validation": self.store.config.enable_h2_validation,
                "max_repair_candidates": self.store.config.max_repair_candidates,
                "enable_provisional_retrieval": self.store.config.enable_provisional_retrieval,
                "identity_recall_mode": self.store.config.identity_recall_mode,
                "enable_mas_guidance": self.store.config.enable_mas_guidance,
                "enable_required_handoffs": self.store.config.enable_required_handoffs,
            },
            "cost": {
                "llm_audit_calls": len(llm_audits),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "llm_latency_seconds": llm_latency,
            },
            "final_test_leakage": leakage,
        }

    def write(self, path: str | Path | None = None) -> Path:
        destination = Path(path) if path else self.store.root / "metrics.json"
        _atomic_write_json(destination, self.compute())
        return destination

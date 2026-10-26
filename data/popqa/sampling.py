import math
import random
from collections import defaultdict
from typing import Any, Iterable


POPQA_ORDERED = "ordered"
POPQA_STRATIFIED_BY_PROP = "stratified_by_prop"
POPQA_SAMPLING_CHOICES = (POPQA_ORDERED, POPQA_STRATIFIED_BY_PROP)


def stratified_prop_indices(
    rows: Iterable[dict[str, Any]],
    sample_size: int,
    *,
    seed: int = 42,
) -> list[int]:
    """Return deterministic prop-stratified indices without looking at answer fields."""
    rows = list(rows)
    sample_size = min(max(0, sample_size), len(rows))
    if sample_size == 0:
        return []

    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        prop = str(row.get("prop") or "__missing_prop__").strip() or "__missing_prop__"
        groups[prop].append(index)

    rng = random.Random(seed)
    for indices in groups.values():
        rng.shuffle(indices)

    quotas = _allocate_prop_quotas({prop: len(indices) for prop, indices in groups.items()}, sample_size)
    selected: list[int] = []
    for prop in sorted(groups):
        selected.extend(groups[prop][: quotas.get(prop, 0)])

    rng.shuffle(selected)
    return selected[:sample_size]


def select_popqa_tasks(
    dataset,
    *,
    num_tasks: int,
    start_from: int = 0,
    sampling: str = POPQA_ORDERED,
    seed: int = 42,
) -> tuple[Any, list[int], dict[str, Any]]:
    """Select PopQA tasks and return the dataset slice plus relative source indices."""
    if sampling not in POPQA_SAMPLING_CHOICES:
        raise ValueError(f"Unsupported PopQA sampling strategy: {sampling}")

    start_from = max(0, start_from)
    num_tasks = max(0, num_tasks)
    target_end = min(start_from + num_tasks, len(dataset))

    if sampling == POPQA_ORDERED:
        indices = list(range(start_from, target_end))
    else:
        # Sample the whole split once so that the first N tasks of any run are a
        # prefix of the full stream (smoke tests see the same questions).
        sampled = stratified_prop_indices(dataset, len(dataset), seed=seed)
        indices = sampled[start_from:target_end]

    info = {
        "sampling": sampling,
        "sample_seed": seed,
        "selected_indices": indices,
    }
    return dataset.select(indices), indices, info


def _allocate_prop_quotas(group_sizes: dict[str, int], sample_size: int) -> dict[str, int]:
    total = sum(group_sizes.values())
    if sample_size >= total:
        return dict(group_sizes)

    props = sorted(group_sizes)
    raw = {prop: group_sizes[prop] * sample_size / total for prop in props}
    quotas = {prop: min(group_sizes[prop], math.floor(raw[prop])) for prop in props}

    if sample_size >= len(props):
        for prop in props:
            if quotas[prop] == 0 and group_sizes[prop] > 0:
                quotas[prop] = 1

    while sum(quotas.values()) > sample_size:
        candidates = [prop for prop in props if quotas[prop] > 0]
        if sample_size >= len(props):
            candidates = [prop for prop in candidates if quotas[prop] > 1]
        prop = min(candidates, key=lambda item: (raw[item] - quotas[item], quotas[item], item))
        quotas[prop] -= 1

    while sum(quotas.values()) < sample_size:
        candidates = [prop for prop in props if quotas[prop] < group_sizes[prop]]
        if not candidates:
            break
        prop = max(candidates, key=lambda item: (raw[item] - quotas[item], group_sizes[item], item))
        quotas[prop] += 1

    return quotas

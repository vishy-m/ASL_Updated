"""Helpers for building compact deterministic pilot manifests."""

from __future__ import annotations

from collections import Counter


def entry_glosses(entry: dict) -> list[str]:
    """Return sorted unique glosses for a manifest entry."""
    return sorted(set(entry.get("glosses", [])))


def select_balanced_subset(entries: list[dict], max_samples: int) -> list[dict]:
    """Cap a split while keeping deterministic label coverage.

    When a split is larger than the pilot budget, a naive head() can erase
    low-frequency pilot labels simply because of sort order. This selector
    keeps ordering deterministic but greedily prioritizes entries that improve
    uncovered or underrepresented labels.
    """
    sorted_entries = sorted(
        entries,
        key=lambda entry: (
            entry.get("split", ""),
            entry.get("id", ""),
            entry.get("features_path", ""),
        ),
    )
    if len(sorted_entries) <= max_samples:
        return sorted_entries

    remaining = list(enumerate(sorted_entries))
    total_counts: Counter[str] = Counter()
    for _, entry in remaining:
        total_counts.update(entry_glosses(entry))

    selected: list[dict] = []
    selected_counts: Counter[str] = Counter()

    while remaining and len(selected) < max_samples:
        best_pos = None
        best_key = None

        for pos, (orig_idx, entry) in enumerate(remaining):
            labels = entry_glosses(entry)
            if not labels:
                continue

            coverage_gain = sum(1 for label in labels if selected_counts[label] == 0)
            balance_gain = sum(1.0 / (1 + selected_counts[label]) for label in labels)
            rarity_gain = sum(1.0 / max(total_counts[label], 1) for label in labels)
            key = (
                coverage_gain,
                balance_gain,
                rarity_gain,
                len(labels),
                -orig_idx,
            )
            if best_key is None or key > best_key:
                best_key = key
                best_pos = pos

        if best_pos is None:
            break

        _, entry = remaining.pop(best_pos)
        selected.append(entry)
        selected_counts.update(entry_glosses(entry))

    if len(selected) < max_samples:
        selected.extend(entry for _, entry in remaining[: max_samples - len(selected)])

    return selected

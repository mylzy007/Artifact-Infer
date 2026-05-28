"""Pure-CPU quality metrics for aggregated MoE routing profiles."""
from __future__ import annotations

import math
from typing import Any


def _pairwise_stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "min": None, "max": None}
    return {
        "mean": float(sum(values) / len(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _row_entropy(row: list[int]) -> float:
    total = float(sum(row))
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in row:
        if count <= 0:
            continue
        p = float(count) / total
        entropy -= p * math.log(p)
    return float(entropy)


def _cosine_similarity(row_a: list[int], row_b: list[int]) -> float:
    dot = float(sum(int(a) * int(b) for a, b in zip(row_a, row_b)))
    norm_a = math.sqrt(float(sum(int(a) * int(a) for a in row_a)))
    norm_b = math.sqrt(float(sum(int(b) * int(b) for b in row_b)))
    if norm_a == 0.0 and norm_b == 0.0:
        return 1.0
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def compute_profile_quality(traffic: list[list[list[int]]]) -> dict[str, Any]:
    per_layer = []
    identical_means = []
    l1_means = []
    cosine_means = []
    all_layers_identical = True

    for layer_id, matrix in enumerate(traffic):
        rows = [[int(x) for x in row] for row in matrix]
        num_rows = len(rows)
        if num_rows == 0:
            per_layer.append(
                {
                    "layer_id": layer_id,
                    "num_src_rows": 0,
                    "all_rows_identical": True,
                    "row_identical_fraction": None,
                    "row_pairwise_l1_distance": _pairwise_stats([]),
                    "row_pairwise_cosine_similarity": _pairwise_stats([]),
                    "row_entropy_per_src": [],
                    "row_entropy_summary": _pairwise_stats([]),
                }
            )
            continue

        base_row = rows[0]
        identical_fraction = sum(1 for row in rows if row == base_row) / num_rows
        all_identical = identical_fraction == 1.0
        if not all_identical:
            all_layers_identical = False

        pairwise_l1 = []
        pairwise_cosine = []
        for i in range(num_rows):
            for j in range(i + 1, num_rows):
                row_i = rows[i]
                row_j = rows[j]
                pairwise_l1.append(
                    float(sum(abs(int(a) - int(b)) for a, b in zip(row_i, row_j)))
                )
                pairwise_cosine.append(_cosine_similarity(row_i, row_j))

        entropies = [_row_entropy(row) for row in rows]
        l1_stats = _pairwise_stats(pairwise_l1)
        cosine_stats = _pairwise_stats(pairwise_cosine)
        entropy_stats = _pairwise_stats(entropies)
        per_layer.append(
            {
                "layer_id": layer_id,
                "num_src_rows": num_rows,
                "all_rows_identical": all_identical,
                "row_identical_fraction": float(identical_fraction),
                "row_pairwise_l1_distance": l1_stats,
                "row_pairwise_cosine_similarity": cosine_stats,
                "row_entropy_per_src": entropies,
                "row_entropy_summary": entropy_stats,
            }
        )
        identical_means.append(float(identical_fraction))
        if l1_stats["mean"] is not None:
            l1_means.append(float(l1_stats["mean"]))
        if cosine_stats["mean"] is not None:
            cosine_means.append(float(cosine_stats["mean"]))

    return {
        "per_layer": per_layer,
        "summary": {
            "num_layers": len(per_layer),
            "row_identical_fraction_mean": (
                float(sum(identical_means) / len(identical_means))
                if identical_means
                else None
            ),
            "row_pairwise_l1_distance_mean": (
                float(sum(l1_means) / len(l1_means))
                if l1_means
                else None
            ),
            "row_pairwise_cosine_similarity_mean": (
                float(sum(cosine_means) / len(cosine_means))
                if cosine_means
                else None
            ),
            "all_layers_identical": bool(all_layers_identical and per_layer),
        },
    }

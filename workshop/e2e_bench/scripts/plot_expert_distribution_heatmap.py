#!/usr/bin/env python3
"""Create vLLM expert-distribution heatmaps and mapping metadata from `.pt` dumps."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

FORWARD_MODE_LABELS = {
    1: "prefill",
    2: "decode",
}


def _normalize_forward_mode(mode: Any) -> int | None:
    if mode is None:
        return None
    try:
        mode_int = int(mode)
    except Exception:
        return None
    # Keep vLLM output aligned with SGLang-facing convention:
    # 1=prefill/extend, 2=decode. Older vLLM dumps may use 3 for mixed.
    if mode_int == 3:
        return 1
    return mode_int


def _normalize_forward_mode_list(values: list[Any]) -> list[int | None]:
    return [_normalize_forward_mode(v) for v in values]


def _read_manifest(path: Path) -> list[Path]:
    paths: list[Path] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            paths.append(Path(line).expanduser().resolve())
    if not paths:
        raise ValueError(f"No record files listed in manifest: {path}")
    return paths


def _resolve_record_path(pt_path: str | None, manifest_path: str | None) -> Path:
    if pt_path is not None:
        path = Path(pt_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Record file not found: {path}")
        return path

    assert manifest_path is not None
    paths = _read_manifest(Path(manifest_path).expanduser().resolve())
    for path in reversed(paths):
        if path.is_file():
            return path
    raise FileNotFoundError(f"No existing record files found in manifest: {manifest_path}")


def _resolve_benchmark_dir(benchmark: str, result_root: str | None) -> Path:
    benchmark_path = Path(benchmark).expanduser()
    if benchmark_path.exists():
        return benchmark_path.resolve()
    if result_root is None:
        raise FileNotFoundError(
            f"Benchmark directory not found: {benchmark}. Provide --result-root or an absolute path."
        )
    path = Path(result_root).expanduser().resolve() / benchmark
    if not path.exists():
        raise FileNotFoundError(f"Benchmark directory not found: {path}")
    return path


def _find_benchmark_case_dirs(benchmark_dir: Path) -> list[Path]:
    manifest_name = "expert_record_files.txt"
    if (benchmark_dir / manifest_name).is_file():
        return [benchmark_dir]
    case_dirs = sorted(
        path for path in benchmark_dir.iterdir() if path.is_dir() and (path / manifest_name).is_file()
    )
    if not case_dirs:
        raise FileNotFoundError(
            f"No case directories with {manifest_name} found under {benchmark_dir}"
        )
    return case_dirs


def _infer_count_divisor(case_dir: Path) -> int:
    run_log = case_dir / "run.log"
    if run_log.is_file():
        text = run_log.read_text(encoding="utf-8", errors="ignore")
        match = re.search(r"num_visible_gpus=(\d+)", text)
        if match:
            value = int(match.group(1))
            if value > 0:
                return value

    server_log = case_dir / "server.log"
    if server_log.is_file():
        text = server_log.read_text(encoding="utf-8", errors="ignore")
        match = re.search(r"tp_size=(\d+)", text)
        if match:
            value = int(match.group(1))
            if value > 0:
                return value

    return 1


def _infer_case_dir_from_source(record_path: Path, manifest_path: str | None) -> Path | None:
    if manifest_path is not None:
        return Path(manifest_path).expanduser().resolve().parent
    parent = record_path.parent
    if parent.name == "expert_records":
        return parent.parent
    return None


def _scale_count_matrix(matrix: np.ndarray, count_divisor: int) -> np.ndarray:
    if count_divisor <= 1:
        return matrix.astype(np.float64, copy=False)
    scaled = matrix.astype(np.float64, copy=False) / float(count_divisor)
    if np.allclose(scaled, np.round(scaled)):
        scaled = np.round(scaled)
    return scaled


def _to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _load_raw_record(path: Path) -> dict[str, Any]:
    data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(data, dict):
        raise ValueError(f"Unsupported record format in {path}: expected dict.")
    if "logical_count" not in data:
        raise ValueError(
            f"Record file {path} does not contain logical_count. "
            "This heatmap script currently supports stat/stat_approx dumps."
        )
    return data


def _extract_segments(data: dict[str, Any]) -> list[dict[str, Any]]:
    raw_segments = data.get("segments")
    if isinstance(raw_segments, list) and raw_segments:
        segments: list[dict[str, Any]] = []
        for raw in raw_segments:
            segment = {
                "segment_index": int(raw.get("segment_index", len(segments))),
                "forward_pass_ids": [int(x) for x in raw.get("forward_pass_ids", [])],
                "forward_modes": _normalize_forward_mode_list(
                    list(raw.get("forward_modes", []))
                ),
                "global_forward_modes": _normalize_forward_mode_list(
                    list(raw.get("global_forward_modes", []))
                ),
                "is_extend_in_batch": list(raw.get("is_extend_in_batch", [])),
                "forward_request_ids": list(raw.get("forward_request_ids", [])),
                "global_physical_count": _to_numpy(raw.get("global_physical_count")),
                "logical_count": _to_numpy(raw.get("logical_count")),
                "rank_count": _to_numpy(raw.get("rank_count")),
                "physical_to_logical_map": _to_numpy(raw.get("physical_to_logical_map")),
                "logical_to_all_physical_map": _to_numpy(
                    raw.get("logical_to_all_physical_map")
                ),
                "logical_to_all_physical_map_num_valid": _to_numpy(
                    raw.get("logical_to_all_physical_map_num_valid")
                ),
                "physical_expert_owner_ep_rank": _to_numpy(
                    raw.get("physical_expert_owner_ep_rank")
                ),
            }
            _validate_segment(segment)
            segments.append(segment)
        return segments

    segment = {
        "segment_index": 0,
        "forward_pass_ids": [int(x) for x in data.get("forward_pass_ids", [])],
        "forward_modes": _normalize_forward_mode_list(
            list(data.get("forward_modes", []))
        ),
        "global_forward_modes": _normalize_forward_mode_list(
            list(data.get("global_forward_modes", []))
        ),
        "is_extend_in_batch": list(data.get("is_extend_in_batch", [])),
        "forward_request_ids": list(data.get("forward_request_ids", [])),
        "global_physical_count": _to_numpy(data.get("global_physical_count")),
        "logical_count": _to_numpy(data.get("logical_count")),
        "rank_count": _to_numpy(data.get("rank_count")),
        "physical_to_logical_map": _to_numpy(data.get("last_physical_to_logical_map")),
        "logical_to_all_physical_map": _to_numpy(data.get("logical_to_all_physical_map")),
        "logical_to_all_physical_map_num_valid": _to_numpy(
            data.get("logical_to_all_physical_map_num_valid")
        ),
        "physical_expert_owner_ep_rank": _to_numpy(
            data.get("physical_expert_owner_ep_rank")
        ),
    }
    _validate_segment(segment)
    return [segment]


def _validate_segment(segment: dict[str, Any]) -> None:
    logical_count = segment["logical_count"]
    if logical_count is None or logical_count.ndim != 3:
        raise ValueError(
            "logical_count must be present as [time, layer, expert], got "
            f"{None if logical_count is None else logical_count.shape}."
        )
    num_steps = int(logical_count.shape[0])
    for key in (
        "forward_pass_ids",
        "forward_modes",
        "global_forward_modes",
        "is_extend_in_batch",
        "forward_request_ids",
    ):
        value = segment.get(key)
        if value and len(value) != num_steps:
            raise ValueError(
                f"{key} must have one entry per time step: expected {num_steps}, got {len(value)}."
            )
    for key in ("global_physical_count", "rank_count"):
        value = segment.get(key)
        if value is not None and value.ndim != 3:
            raise ValueError(f"{key} must be 3D when present, got {value.shape}.")
    physical_to_logical_map = segment.get("physical_to_logical_map")
    if physical_to_logical_map is not None and physical_to_logical_map.ndim != 2:
        raise ValueError(
            "physical_to_logical_map must be [layer, physical_expert], got "
            f"{physical_to_logical_map.shape}."
        )
    physical_owner = segment.get("physical_expert_owner_ep_rank")
    if physical_owner is not None and physical_owner.ndim != 2:
        raise ValueError(
            "physical_expert_owner_ep_rank must be [layer, physical_expert], got "
            f"{physical_owner.shape}."
        )


def _maybe_load_model_config(model_name: str | None) -> dict[str, Any] | None:
    if not model_name:
        return None

    path = Path(model_name).expanduser()
    candidates: list[Path] = []
    if path.is_dir():
        candidates.append(path / "config.json")
    elif path.is_file():
        candidates.append(path)

    for candidate in candidates:
        try:
            if candidate.is_file():
                return json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
    return None


def _infer_active_layer_ids(segments: list[dict[str, Any]]) -> list[int]:
    if not segments:
        return []
    activity = np.zeros(
        (segments[0]["logical_count"].shape[1],),
        dtype=np.int64,
    )
    for segment in segments:
        activity += segment["logical_count"].sum(axis=(0, 2))
    return np.flatnonzero(activity > 0).astype(np.int64).tolist()


def _resolve_moe_layer_ids(
    num_layers: int,
    segments: list[dict[str, Any]],
    model_name: str | None,
) -> list[int]:
    config = _maybe_load_model_config(model_name)
    if config is not None:
        first_k_dense_replace = config.get("first_k_dense_replace")
        moe_layer_freq = config.get("moe_layer_freq")
        has_moe = any(
            config.get(key) is not None
            for key in ("n_routed_experts", "num_experts", "num_local_experts")
        )
        if (
            has_moe
            and isinstance(first_k_dense_replace, int)
            and isinstance(moe_layer_freq, int)
            and first_k_dense_replace >= 0
            and moe_layer_freq > 0
        ):
            moe_layer_ids = [
                layer_id
                for layer_id in range(num_layers)
                if layer_id >= first_k_dense_replace
                and (layer_id - first_k_dense_replace) % moe_layer_freq == 0
            ]
            if moe_layer_ids:
                print(
                    "Using MoE layer ids from model config: "
                    f"{moe_layer_ids[:8]}{'...' if len(moe_layer_ids) > 8 else ''}"
                )
                return moe_layer_ids

    active_layer_ids = _infer_active_layer_ids(segments)
    if active_layer_ids and len(active_layer_ids) < num_layers:
        print(
            "Falling back to active-layer inference for MoE layers: "
            f"{active_layer_ids[:8]}{'...' if len(active_layer_ids) > 8 else ''}"
        )
        return active_layer_ids

    return list(range(num_layers))


def _infer_ep_size(data: dict[str, Any], segments: list[dict[str, Any]]) -> int:
    if data.get("ep_size") is not None:
        return int(data["ep_size"])
    for segment in segments:
        rank_count = segment.get("rank_count")
        if rank_count is not None:
            return int(rank_count.shape[2])
        physical_owner = segment.get("physical_expert_owner_ep_rank")
        if physical_owner is not None and physical_owner.size > 0:
            return int(physical_owner.max()) + 1
    raise ValueError("Unable to infer ep_size from record.")


def _logical_owner_ep_ranks(
    physical_to_logical_map: np.ndarray,
    physical_owner: np.ndarray,
    num_logical_experts: int,
) -> list[list[list[int]]]:
    owners_by_layer: list[list[list[int]]] = []
    for layer_idx in range(physical_to_logical_map.shape[0]):
        owners: list[set[int]] = [set() for _ in range(num_logical_experts)]
        for physical_expert_id, logical_expert_id in enumerate(
            physical_to_logical_map[layer_idx].tolist()
        ):
            owners[int(logical_expert_id)].add(
                int(physical_owner[layer_idx, physical_expert_id])
            )
        owners_by_layer.append([sorted(x) for x in owners])
    return owners_by_layer


def _compute_rank_matrix_for_segment(
    segment: dict[str, Any], ep_size: int
) -> tuple[np.ndarray | None, bool]:
    global_physical_count = segment.get("global_physical_count")
    physical_owner = segment.get("physical_expert_owner_ep_rank")
    if global_physical_count is not None and physical_owner is not None:
        rank_matrix = np.zeros(
            (
                global_physical_count.shape[0],
                global_physical_count.shape[1],
                ep_size,
            ),
            dtype=np.int64,
        )
        for ep_rank in range(ep_size):
            mask = (physical_owner == ep_rank)[None, :, :]
            rank_matrix[:, :, ep_rank] = np.where(mask, global_physical_count, 0).sum(
                axis=2
            )
        return rank_matrix, True

    rank_count = segment.get("rank_count")
    if rank_count is not None:
        return rank_count.astype(np.int64, copy=False), False

    return None, False


def _mapping_check_matches(
    logical_count: np.ndarray,
    mapping_rank_count: np.ndarray | None,
    physical_to_logical_map: np.ndarray | None,
    physical_owner: np.ndarray | None,
    ep_size: int,
) -> bool | None:
    if mapping_rank_count is None:
        return None
    if physical_to_logical_map is None or physical_owner is None:
        return None

    num_logical_experts = logical_count.shape[2]
    owners_by_layer = _logical_owner_ep_ranks(
        physical_to_logical_map, physical_owner, num_logical_experts
    )
    if any(len(owner_list) != 1 for layer in owners_by_layer for owner_list in layer):
        return None

    logical_rank_count = np.zeros_like(mapping_rank_count)
    for layer_idx, owners in enumerate(owners_by_layer):
        owner_index = np.asarray([owner_list[0] for owner_list in owners], dtype=np.int64)
        for ep_rank in range(ep_size):
            logical_rank_count[:, layer_idx, ep_rank] = logical_count[:, layer_idx, owner_index == ep_rank].sum(
                axis=1
            )
    return bool(np.array_equal(logical_rank_count, mapping_rank_count))


def _normalize_rows_to_percent(matrix: np.ndarray) -> np.ndarray:
    plot_data = matrix.astype(np.float64, copy=False)
    row_sums = plot_data.sum(axis=1, keepdims=True)
    return np.divide(
        plot_data * 100.0,
        row_sums,
        out=np.zeros_like(plot_data, dtype=np.float64),
        where=row_sums > 0,
    )


def _safe_vmax(plot_data: np.ndarray, vmax_percentile: float) -> float:
    if plot_data.size == 0:
        return 1.0
    vmax = float(np.percentile(plot_data, vmax_percentile))
    if vmax <= 0:
        vmax = max(float(plot_data.max()), 1.0)
    return vmax


def _safe_integer_vmax(plot_data: np.ndarray, vmax_percentile: float) -> float:
    vmax = _safe_vmax(plot_data, vmax_percentile)
    return float(max(int(np.ceil(vmax)), 1))


def _segment_start_ticks(time_indices: np.ndarray) -> list[tuple[int, int]]:
    if time_indices.size == 0:
        return []
    ticks: list[tuple[int, int]] = [(0, int(time_indices[0]))]
    prev = int(time_indices[0])
    for row_index, time_index in enumerate(time_indices[1:], start=1):
        current = int(time_index)
        if current != prev + 1:
            ticks.append((row_index, current))
        prev = current
    return ticks


def _save_csv(
    path: Path,
    matrix: np.ndarray,
    prefix: str,
    time_indices: np.ndarray,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        header = ["time_index"] + [f"{prefix}_{i}" for i in range(matrix.shape[1])]
        writer.writerow(header)
        for row_index, row in enumerate(matrix):
            serialized_row = []
            for value in row.tolist():
                if isinstance(value, float) and value.is_integer():
                    serialized_row.append(int(value))
                else:
                    serialized_row.append(value)
            writer.writerow([int(time_indices[row_index]), *serialized_row])


def _save_row_index_map(
    path: Path,
    *,
    time_indices: np.ndarray,
    forward_modes: np.ndarray,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["row_index", "time_index", "forward_mode", "forward_mode_name"])
        for row_index, time_index in enumerate(time_indices.tolist()):
            mode = (
                _normalize_forward_mode(forward_modes[row_index])
                if row_index < len(forward_modes)
                else None
            )
            mode_value = int(mode) if mode is not None else -1
            writer.writerow(
                [
                    row_index,
                    int(time_index),
                    mode_value,
                    FORWARD_MODE_LABELS.get(
                        mode_value, "other" if mode_value >= 0 else ""
                    ),
                ]
            )


def _plot_dual_heatmap(
    matrix: np.ndarray,
    x_label: str,
    title: str,
    out_path: Path,
    vmax_percentile: float,
    *,
    time_indices: np.ndarray,
    include_count_panel: bool = True,
) -> None:
    share_data = _normalize_rows_to_percent(matrix)
    share_vmax = _safe_vmax(share_data, vmax_percentile)
    share_normalized = np.clip(
        share_data / max(share_vmax, 1e-12), a_min=0.0, a_max=1.0
    )
    count_data = matrix.astype(np.float64, copy=False)
    count_vmax = _safe_integer_vmax(count_data, vmax_percentile)
    count_normalized = np.clip(
        count_data / max(count_vmax, 1e-12), a_min=0.0, a_max=1.0
    )

    num_rows, num_cols = share_normalized.shape
    heatmap_width = min(max(num_cols * 12, 320), 900)
    heatmap_height = min(max(num_rows * 2, 240), 900)
    left_margin = 84
    right_margin = 110
    top_margin = 54
    bottom_margin = 72
    colorbar_width = 20
    panel_gap = 72
    panel_width = heatmap_width + 24 + colorbar_width
    panel_count = 2 if include_count_panel else 1
    canvas_width = (
        left_margin
        + panel_width * panel_count
        + (panel_gap if include_count_panel else 0)
        + right_margin
    )
    canvas_height = top_margin + heatmap_height + bottom_margin

    canvas = Image.new("RGB", (canvas_width, canvas_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    resampling = getattr(Image, "Resampling", Image)

    panel_specs = [("Token share (%)", share_normalized, share_vmax, left_margin)]
    if include_count_panel:
        panel_specs.append(
            (
                "Token count",
                count_normalized,
                count_vmax,
                left_margin + panel_width + panel_gap,
            )
        )
    y_ticks = _segment_start_ticks(time_indices)

    _draw_centered_text(draw, canvas_width // 2, 16, title, font=font, fill=(0, 0, 0))
    draw.text((14, 14), "Time index", font=font, fill=(0, 0, 0))

    for panel_title, normalized, vmax, panel_left in panel_specs:
        heatmap = Image.fromarray(_apply_colormap(normalized)).resize(
            (heatmap_width, heatmap_height),
            resample=resampling.NEAREST,
        )
        heatmap_origin = (panel_left, top_margin)
        canvas.paste(heatmap, heatmap_origin)
        draw.rectangle(
            (
                heatmap_origin[0] - 1,
                heatmap_origin[1] - 1,
                heatmap_origin[0] + heatmap_width,
                heatmap_origin[1] + heatmap_height,
            ),
            outline=(90, 90, 90),
            width=1,
        )

        colorbar = Image.fromarray(
            _apply_colormap(
                np.linspace(1.0, 0.0, heatmap_height, dtype=np.float64).reshape(-1, 1)
            )
        ).resize((colorbar_width, heatmap_height), resample=resampling.NEAREST)
        colorbar_x = panel_left + heatmap_width + 24
        canvas.paste(colorbar, (colorbar_x, top_margin))
        draw.rectangle(
            (
                colorbar_x - 1,
                top_margin - 1,
                colorbar_x + colorbar_width,
                top_margin + heatmap_height,
            ),
            outline=(90, 90, 90),
            width=1,
        )

        _draw_centered_text(
            draw,
            panel_left + heatmap_width // 2,
            canvas_height - 24,
            x_label,
            font=font,
            fill=(0, 0, 0),
        )
        _draw_centered_text(
            draw,
            panel_left + heatmap_width // 2,
            32,
            panel_title,
            font=font,
            fill=(0, 0, 0),
        )
        _draw_axis_ticks(
            draw=draw,
            font=font,
            start=panel_left,
            length=heatmap_width,
            axis_values=num_cols,
            baseline=top_margin + heatmap_height + 8,
            vertical=False,
        )
        _draw_axis_ticks(
            draw=draw,
            font=font,
            start=top_margin,
            length=heatmap_height,
            axis_values=num_rows,
            baseline=panel_left - 8,
            vertical=True,
            custom_ticks=y_ticks,
        )
        _draw_colorbar_ticks(
            draw=draw,
            font=font,
            x=colorbar_x + colorbar_width + 8,
            y=top_margin,
            height=heatmap_height,
            vmax=vmax,
            integer_values=(panel_title == "Token count"),
        )
    canvas.save(out_path)


def _apply_colormap(normalized: np.ndarray) -> np.ndarray:
    stops = [
        (0.0, np.array((255, 255, 255), dtype=np.float64)),
        (0.15, np.array((255, 245, 204), dtype=np.float64)),
        (0.35, np.array((254, 224, 139), dtype=np.float64)),
        (0.60, np.array((253, 174, 97), dtype=np.float64)),
        (0.80, np.array((244, 109, 67), dtype=np.float64)),
        (1.0, np.array((165, 0, 38), dtype=np.float64)),
    ]
    clipped = np.clip(normalized, 0.0, 1.0)
    flat = clipped.reshape(-1)
    colors = np.zeros((flat.size, 3), dtype=np.float64)
    for (left_t, left_color), (right_t, right_color) in zip(stops, stops[1:]):
        mask = (flat >= left_t) & (flat <= right_t)
        if not np.any(mask):
            continue
        scale = (flat[mask] - left_t) / max(right_t - left_t, 1e-12)
        colors[mask] = left_color + (right_color - left_color) * scale[:, None]
    colors[flat <= stops[0][0]] = stops[0][1]
    colors[flat >= stops[-1][0]] = stops[-1][1]
    return colors.reshape(*clipped.shape, 3).astype(np.uint8)


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    center_x: int,
    y: int,
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    draw.text((center_x - (right - left) / 2, y), text, font=font, fill=fill)


def _iter_tick_values(axis_values: int) -> list[int]:
    if axis_values <= 1:
        return [0]
    return sorted({0, axis_values // 2, axis_values - 1})


def _draw_axis_ticks(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    start: int,
    length: int,
    axis_values: int,
    baseline: int,
    vertical: bool,
    tick_labels: list[int] | None = None,
    custom_ticks: list[tuple[int, int]] | None = None,
) -> None:
    denom = max(axis_values - 1, 1)
    if custom_ticks is not None:
        ticks = custom_ticks
    else:
        ticks = []
        for tick in _iter_tick_values(axis_values):
            if tick_labels is not None and 0 <= tick < len(tick_labels):
                label_value = tick_labels[tick]
            else:
                label_value = tick
            ticks.append((tick, label_value))
    for tick, label_value in ticks:
        offset = int(round(tick / denom * max(length - 1, 0)))
        label = str(label_value)
        if vertical:
            y = start + offset
            left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
            draw.line(
                (baseline + 2, y, baseline + 6, y),
                fill=(90, 90, 90),
                width=1,
            )
            draw.text(
                (baseline - (right - left), y - (bottom - top) / 2),
                label,
                font=font,
                fill=(0, 0, 0),
            )
        else:
            x = start + offset
            _draw_centered_text(draw, x, baseline, label, font=font, fill=(0, 0, 0))


def _draw_colorbar_ticks(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    x: int,
    y: int,
    height: int,
    vmax: float,
    integer_values: bool = False,
) -> None:
    for ratio, value in ((0.0, vmax), (0.5, vmax / 2), (1.0, 0.0)):
        if integer_values:
            if abs(value - round(value)) < 1e-6:
                label = str(int(round(value)))
            else:
                label = f"{value:.1f}"
        else:
            label = f"{value:.1f}"
        ypos = y + int(round(ratio * max(height - 1, 0)))
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        draw.text((x, ypos - (bottom - top) / 2), label, font=font, fill=(0, 0, 0))


def _parse_layers(layer: int, num_layers: int) -> list[int]:
    if layer == -1:
        return list(range(num_layers))
    if layer < 0 or layer >= num_layers:
        raise ValueError(f"Layer {layer} out of range [0, {num_layers - 1}]")
    return [layer]


def _export_mapping_jsonl(
    path: Path,
    segments: list[dict[str, Any]],
    ep_size: int,
    num_logical_experts: int,
    model_name: str | None,
    moe_layer_ids_in_model: list[int],
) -> tuple[bool, bool]:
    has_records = False
    has_dynamic_records = False
    time_index_start = 0

    with path.open("w", encoding="utf-8") as f:
        for segment in segments:
            physical_to_logical_map = segment.get("physical_to_logical_map")
            physical_owner = segment.get("physical_expert_owner_ep_rank")
            if physical_to_logical_map is None or physical_owner is None:
                time_index_start += int(segment["logical_count"].shape[0])
                continue

            owners_by_layer = _logical_owner_ep_ranks(
                physical_to_logical_map, physical_owner, num_logical_experts
            )
            num_steps = int(segment["logical_count"].shape[0])
            forward_pass_ids = segment.get("forward_pass_ids") or []
            record_type = "ep_static_map" if segment["segment_index"] == 0 else "ep_dynamic_map"
            event = "model_init" if segment["segment_index"] == 0 else "eplb_rebalance"
            if segment["segment_index"] > 0:
                has_dynamic_records = True

            for layer_idx, model_layer_id in enumerate(moe_layer_ids_in_model):
                global_owner = owners_by_layer[model_layer_id]
                global_owner_first = [
                    owners[0] if len(owners) == 1 else (-1 if not owners else owners[0])
                    for owners in global_owner
                ]
                has_replicas = any(len(owners) > 1 for owners in global_owner)

                for ep_rank in range(ep_size):
                    local_physical_experts = np.where(
                        physical_owner[model_layer_id] == ep_rank
                    )[0].tolist()
                    local_logical_experts = [
                        int(physical_to_logical_map[model_layer_id, physical_id])
                        for physical_id in local_physical_experts
                    ]

                    record = {
                        "record_type": record_type,
                        "event": event,
                        "segment_index": int(segment["segment_index"]),
                        "time_index_start": int(time_index_start),
                        "time_index_end_exclusive": int(time_index_start + num_steps),
                        "forward_pass_id_start": (
                            int(forward_pass_ids[0]) if forward_pass_ids else None
                        ),
                        "forward_pass_id_end": (
                            int(forward_pass_ids[-1]) if forward_pass_ids else None
                        ),
                        "layer_id": int(layer_idx),
                        "layer_name": f"moe_layer_{layer_idx}",
                        "model_layer_id": int(model_layer_id),
                        "model_name": model_name,
                        "ep_rank": int(ep_rank),
                        "ep_size": int(ep_size),
                        "local_num_experts": int(len(local_physical_experts)),
                        "global_num_experts": int(num_logical_experts),
                        "local_physical_experts": local_physical_experts,
                        "local_logical_experts": local_logical_experts,
                        "local_logical_experts_unique": sorted(set(local_logical_experts)),
                        "global_logical_owner_ep_rank": global_owner_first,
                        "global_logical_owner_ep_ranks": global_owner,
                        "has_replicated_logical_experts": bool(has_replicas),
                    }
                    f.write(json.dumps(record, ensure_ascii=True) + "\n")
                    has_records = True

            time_index_start += num_steps

    return has_records, has_dynamic_records


def _export_forward_steps_jsonl(
    path: Path,
    segments: list[dict[str, Any]],
    *,
    skip_initial_steps: int,
) -> bool:
    has_records = False
    skipped = 0
    time_index = 0

    with path.open("w", encoding="utf-8") as f:
        for segment in segments:
            num_steps = int(segment["logical_count"].shape[0])
            forward_pass_ids = segment.get("forward_pass_ids") or []
            forward_modes = segment.get("forward_modes") or []
            global_forward_modes = segment.get("global_forward_modes") or []
            is_extend_in_batch = segment.get("is_extend_in_batch") or []
            forward_request_ids = segment.get("forward_request_ids") or []

            for step_idx in range(num_steps):
                if skipped < skip_initial_steps:
                    skipped += 1
                    continue

                record = {
                    "time_index": int(time_index),
                    "segment_index": int(segment["segment_index"]),
                    "step_index_in_segment": int(step_idx),
                    "forward_pass_id": (
                        int(forward_pass_ids[step_idx])
                        if step_idx < len(forward_pass_ids)
                        else None
                    ),
                    "forward_mode": (
                        _normalize_forward_mode(forward_modes[step_idx])
                        if step_idx < len(forward_modes)
                        else None
                    ),
                    "global_forward_mode": (
                        _normalize_forward_mode(global_forward_modes[step_idx])
                        if step_idx < len(global_forward_modes)
                        else None
                    ),
                    "is_extend_in_batch": (
                        bool(is_extend_in_batch[step_idx])
                        if step_idx < len(is_extend_in_batch)
                        else None
                    ),
                }
                if step_idx < len(forward_request_ids):
                    request_ids = forward_request_ids[step_idx]
                    if request_ids:
                        record["request_ids"] = list(request_ids)

                f.write(json.dumps(record, ensure_ascii=True) + "\n")
                has_records = True
                time_index += 1

    return has_records


def _write_record_meta(
    path: Path,
    *,
    case_name: str | None,
    record_path: Path,
    manifest_path: str | None,
    heatmap_dir: Path,
    mapping_jsonl: Path | None,
    forward_steps_jsonl: Path | None,
    num_segments: int,
    num_layers: int,
    moe_layer_ids_in_model: list[int],
    num_logical_experts: int,
    num_physical_experts: int | None,
    ep_size: int,
    has_mapping_records: bool,
    has_dynamic_mapping_records: bool,
    has_forward_step_records: bool,
    rank_mapping_check_passed: bool | None,
    enable_eplb: str | None,
    count_divisor: int,
) -> None:
    meta = {
        "case_name": case_name,
        "expert_record_pt": str(record_path),
        "expert_record_manifest": manifest_path,
        "heatmap_dir": str(heatmap_dir),
        "mapping_jsonl": str(mapping_jsonl) if mapping_jsonl is not None else None,
        "forward_steps_jsonl": (
            str(forward_steps_jsonl) if forward_steps_jsonl is not None else None
        ),
        "num_segments": int(num_segments),
        "eplb_num_layers": int(num_layers),
        "moe_layer_ids_in_model": [int(x) for x in moe_layer_ids_in_model],
        "layer_id_is_moe_layer_index": True,
        "num_logical_experts": int(num_logical_experts),
        "num_physical_experts": (
            int(num_physical_experts) if num_physical_experts is not None else None
        ),
        "ep_size": int(ep_size),
        "has_mapping_records": bool(has_mapping_records),
        "has_dynamic_mapping_records": bool(has_dynamic_mapping_records),
        "has_forward_step_records": bool(has_forward_step_records),
        "rank_mapping_check_passed": rank_mapping_check_passed,
        "enable_eplb": enable_eplb,
        "count_divisor": int(count_divisor),
    }
    path.write_text(json.dumps(meta, ensure_ascii=True, indent=2), encoding="utf-8")


def _process_record(
    *,
    record_path: Path,
    manifest_path: str | None,
    out_dir: Path,
    mapping_jsonl_path: Path | None,
    record_meta_path: Path | None,
    forward_steps_jsonl_path: Path | None,
    layer: int,
    skip_initial_steps: int,
    vmax_percentile: float,
    case_name: str | None,
    model_name: str | None,
    enable_eplb: str | None,
    count_divisor: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    data = _load_raw_record(record_path)
    segments = _extract_segments(data)
    ep_size = _infer_ep_size(data, segments)
    num_model_layers = int(segments[0]["logical_count"].shape[1])
    moe_layer_ids_in_model = _resolve_moe_layer_ids(
        num_layers=num_model_layers,
        segments=segments,
        model_name=model_name,
    )
    num_layers = len(moe_layer_ids_in_model)
    num_logical_experts = int(segments[0]["logical_count"].shape[2])
    num_physical_experts = next(
        (
            int(segment["global_physical_count"].shape[2])
            for segment in segments
            if segment.get("global_physical_count") is not None
        ),
        int(data["num_physical_experts"]) if data.get("num_physical_experts") else None,
    )

    physical_expert_chunks: list[np.ndarray] = []
    logical_expert_chunks: list[np.ndarray] = []
    rank_chunks: list[np.ndarray] = []
    forward_mode_chunks: list[np.ndarray] = []
    rank_mapping_check_passed: bool | None = True
    missing_rank_matrix = False
    missing_physical_expert_matrix = False
    for segment in segments:
        logical_count = segment["logical_count"][
            :, moe_layer_ids_in_model, :
        ].astype(np.int64, copy=False)
        global_physical_count = segment.get("global_physical_count")
        if global_physical_count is None:
            missing_physical_expert_matrix = True
        else:
            global_physical_count = global_physical_count[:, moe_layer_ids_in_model, :]
            physical_expert_chunks.append(
                global_physical_count.astype(np.int64, copy=False)
            )
        rank_count, used_mapping = _compute_rank_matrix_for_segment(segment, ep_size)
        if rank_count is not None:
            rank_count = rank_count[:, moe_layer_ids_in_model, :]
        physical_to_logical_map = segment.get("physical_to_logical_map")
        if physical_to_logical_map is not None:
            physical_to_logical_map = physical_to_logical_map[moe_layer_ids_in_model, :]
        physical_owner = segment.get("physical_expert_owner_ep_rank")
        if physical_owner is not None:
            physical_owner = physical_owner[moe_layer_ids_in_model, :]
        raw_forward_modes = segment.get("forward_modes") or []
        if len(raw_forward_modes) == logical_count.shape[0]:
            normalized_modes = (_normalize_forward_mode(mode) for mode in raw_forward_modes)
            forward_mode_chunk = np.asarray(
                [int(mode) if mode is not None else -1 for mode in normalized_modes],
                dtype=np.int64,
            )
        else:
            forward_mode_chunk = np.full((logical_count.shape[0],), -1, dtype=np.int64)
        mapping_check = _mapping_check_matches(
            logical_count=logical_count,
            mapping_rank_count=rank_count,
            physical_to_logical_map=physical_to_logical_map,
            physical_owner=physical_owner,
            ep_size=ep_size,
        )
        if mapping_check is False:
            rank_mapping_check_passed = False
        elif mapping_check is None and rank_mapping_check_passed is True:
            rank_mapping_check_passed = None
        logical_expert_chunks.append(logical_count)
        if rank_count is None:
            missing_rank_matrix = True
        else:
            rank_chunks.append(rank_count.astype(np.int64, copy=False))
        forward_mode_chunks.append(forward_mode_chunk)
        if used_mapping:
            print(
                "Computed rank matrix from global physical counts grouped by physical expert owner GPU "
                f"for segment {segment['segment_index']}."
            )

    physical_expert_matrix_all = (
        np.concatenate(physical_expert_chunks, axis=0)
        if not missing_physical_expert_matrix and physical_expert_chunks
        else None
    )
    logical_expert_matrix_all = (
        np.concatenate(logical_expert_chunks, axis=0)
        if logical_expert_chunks
        else np.zeros((0, num_layers, num_logical_experts), dtype=np.int64)
    )
    rank_matrix_all = None
    if not missing_rank_matrix and rank_chunks:
        rank_matrix_all = np.concatenate(rank_chunks, axis=0)
    elif missing_rank_matrix:
        print(
            "Rank heatmaps skipped because this record does not contain "
            "global_physical_count/rank_count for every segment."
        )
    forward_modes_all = (
        np.concatenate(forward_mode_chunks, axis=0)
        if forward_mode_chunks
        else np.zeros((0,), dtype=np.int64)
    )
    time_indices_all = np.arange(logical_expert_matrix_all.shape[0], dtype=np.int64)

    if skip_initial_steps > 0:
        if skip_initial_steps >= logical_expert_matrix_all.shape[0]:
            raise ValueError(
                f"--skip-initial-steps={skip_initial_steps} removes all "
                f"{logical_expert_matrix_all.shape[0]} time indices."
            )
        if physical_expert_matrix_all is not None:
            physical_expert_matrix_all = physical_expert_matrix_all[skip_initial_steps:]
        logical_expert_matrix_all = logical_expert_matrix_all[skip_initial_steps:]
        if rank_matrix_all is not None:
            rank_matrix_all = rank_matrix_all[skip_initial_steps:]
        forward_modes_all = forward_modes_all[skip_initial_steps:]
        time_indices_all = time_indices_all[skip_initial_steps:]

    selected_layers = _parse_layers(layer, num_layers)
    mode_variants: list[tuple[str, np.ndarray, str]] = [
        ("all", np.ones_like(forward_modes_all, dtype=bool), "All"),
        ("prefill", forward_modes_all == 1, "Prefill/EXTEND"),
        ("decode", forward_modes_all == 2, "Decode"),
    ]
    for layer_idx in selected_layers:
        for mode_suffix, mode_mask, mode_title in mode_variants:
            if not np.any(mode_mask):
                continue

            logical_expert_matrix = logical_expert_matrix_all[mode_mask, layer_idx, :]
            logical_expert_matrix_scaled = _scale_count_matrix(
                logical_expert_matrix, count_divisor
            )
            variant_time_indices = time_indices_all[mode_mask]
            variant_forward_modes = forward_modes_all[mode_mask]
            if mode_suffix == "all":
                row_map_csv = out_dir / f"row_index_map_layer{layer_idx}.csv"
                logical_expert_csv = out_dir / f"logical_expert_load_layer{layer_idx}.csv"
                logical_expert_png = out_dir / f"logical_expert_load_layer{layer_idx}.png"
            else:
                row_map_csv = out_dir / f"row_index_map_{mode_suffix}_layer{layer_idx}.csv"
                logical_expert_csv = (
                    out_dir / f"logical_expert_load_{mode_suffix}_layer{layer_idx}.csv"
                )
                logical_expert_png = (
                    out_dir / f"logical_expert_load_{mode_suffix}_layer{layer_idx}.png"
                )

            _save_row_index_map(
                row_map_csv,
                time_indices=variant_time_indices,
                forward_modes=variant_forward_modes,
            )

            _save_csv(
                logical_expert_csv,
                logical_expert_matrix_scaled,
                "logical_expert",
                variant_time_indices,
            )
            _plot_dual_heatmap(
                logical_expert_matrix_scaled,
                "Logical expert id",
                f"{mode_title} Logical Expert Load Heatmap (layer={layer_idx}, count/{count_divisor})",
                logical_expert_png,
                vmax_percentile=vmax_percentile,
                time_indices=variant_time_indices,
                include_count_panel=(mode_suffix != "all"),
            )

            print(f"Saved: {logical_expert_csv}")
            print(f"Saved: {logical_expert_png}")
            print(f"Saved: {row_map_csv}")

            if rank_matrix_all is not None:
                rank_matrix = rank_matrix_all[mode_mask, layer_idx, :]
                rank_matrix_scaled = _scale_count_matrix(rank_matrix, count_divisor)
                if mode_suffix == "all":
                    rank_csv = out_dir / f"rank_load_layer{layer_idx}.csv"
                    rank_png = out_dir / f"rank_load_layer{layer_idx}.png"
                else:
                    rank_csv = out_dir / f"rank_load_{mode_suffix}_layer{layer_idx}.csv"
                    rank_png = out_dir / f"rank_load_{mode_suffix}_layer{layer_idx}.png"

                _save_csv(
                    rank_csv,
                    rank_matrix_scaled,
                    "rank",
                    variant_time_indices,
                )
                _plot_dual_heatmap(
                    rank_matrix_scaled,
                    "EP rank",
                    f"{mode_title} EP Rank Load Heatmap (layer={layer_idx}, count/{count_divisor})",
                    rank_png,
                    vmax_percentile=vmax_percentile,
                    time_indices=variant_time_indices,
                    include_count_panel=(mode_suffix != "all"),
                )

                print(f"Saved: {rank_csv}")
                print(f"Saved: {rank_png}")

    has_mapping_records = False
    has_dynamic_mapping_records = False
    if mapping_jsonl_path is not None:
        mapping_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        has_mapping_records, has_dynamic_mapping_records = _export_mapping_jsonl(
            mapping_jsonl_path,
            segments=segments,
            ep_size=ep_size,
            num_logical_experts=num_logical_experts,
            model_name=model_name,
            moe_layer_ids_in_model=moe_layer_ids_in_model,
        )
        print(f"Saved: {mapping_jsonl_path}")

    has_forward_step_records = False
    if forward_steps_jsonl_path is not None:
        forward_steps_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        has_forward_step_records = _export_forward_steps_jsonl(
            forward_steps_jsonl_path,
            segments=segments,
            skip_initial_steps=skip_initial_steps,
        )
        print(f"Saved: {forward_steps_jsonl_path}")

    if record_meta_path is not None:
        record_meta_path.parent.mkdir(parents=True, exist_ok=True)
        _write_record_meta(
            record_meta_path,
            case_name=case_name,
            record_path=record_path,
            manifest_path=manifest_path,
            heatmap_dir=out_dir,
            mapping_jsonl=mapping_jsonl_path,
            forward_steps_jsonl=forward_steps_jsonl_path,
            num_segments=len(segments),
            num_layers=num_layers,
            moe_layer_ids_in_model=moe_layer_ids_in_model,
            num_logical_experts=num_logical_experts,
            num_physical_experts=num_physical_experts,
            ep_size=ep_size,
            has_mapping_records=has_mapping_records,
            has_dynamic_mapping_records=has_dynamic_mapping_records,
            has_forward_step_records=has_forward_step_records,
            rank_mapping_check_passed=rank_mapping_check_passed,
            enable_eplb=enable_eplb,
            count_divisor=count_divisor,
        )
        print(f"Saved: {record_meta_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate vLLM expert/rank load heatmaps plus mapping metadata."
    )
    source_group = parser.add_mutually_exclusive_group(required=False)
    source_group.add_argument("--pt", default=None, help="Path to a single .pt record file.")
    source_group.add_argument(
        "--manifest",
        default=None,
        help="Path to expert_record_files.txt generated by the matrix script.",
    )
    source_group.add_argument(
        "--benchmark",
        default=None,
        help="Benchmark directory name under --result-root, or an absolute benchmark directory path.",
    )
    parser.add_argument(
        "--result-root",
        default="/home/lzy/eval/vllm_deepseek_v2_lite_matrix4",
        help="Root directory that contains benchmark result directories.",
    )
    parser.add_argument("--out-dir", required=False, help="Directory for CSV/PNG outputs.")
    parser.add_argument(
        "--mapping-jsonl",
        default=None,
        help="Optional output path for expert/GPU mapping JSONL.",
    )
    parser.add_argument(
        "--record-meta-json",
        default=None,
        help="Optional output path for record metadata JSON.",
    )
    parser.add_argument(
        "--forward-steps-jsonl",
        default=None,
        help="Optional output path for per-forward step metadata JSONL.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=-1,
        help="MoE layer index to plot. Use -1 to export all layers.",
    )
    parser.add_argument(
        "--skip-initial-steps",
        type=int,
        default=0,
        help="Skip the first N time indices before exporting.",
    )
    parser.add_argument(
        "--vmax-percentile",
        type=float,
        default=99.0,
        help="Color vmax percentile for robust heatmap scaling.",
    )
    parser.add_argument("--case-name", default=None, help="Optional case name for metadata.")
    parser.add_argument("--model-name", default=None, help="Optional model name for metadata.")
    parser.add_argument(
        "--enable-eplb",
        default=None,
        help="Optional string flag copied into record metadata.",
    )
    parser.add_argument(
        "--count-divisor",
        type=int,
        default=None,
        help="Optional divisor applied to count matrices before CSV/plot export. Defaults to auto-inferred GPU count.",
    )
    args = parser.parse_args()
    if not any((args.pt, args.manifest, args.benchmark)):
        parser.error("one of --pt, --manifest, or --benchmark is required")

    if args.benchmark is not None:
        benchmark_dir = _resolve_benchmark_dir(args.benchmark, args.result_root)
        case_dirs = _find_benchmark_case_dirs(benchmark_dir)
        for case_dir in case_dirs:
            print(f"Processing case: {case_dir}")
            case_name = case_dir.name
            count_divisor = args.count_divisor or _infer_count_divisor(case_dir)
            manifest_path = str(case_dir / "expert_record_files.txt")
            try:
                record_path = _resolve_record_path(None, manifest_path)
            except (FileNotFoundError, ValueError) as exc:
                print(f"Skipping case {case_dir}: {exc}")
                continue

            _process_record(
                record_path=record_path,
                manifest_path=manifest_path,
                out_dir=case_dir / "heatmap",
                mapping_jsonl_path=case_dir / "expert_mapping.jsonl",
                record_meta_path=case_dir / "record_meta.json",
                forward_steps_jsonl_path=case_dir / "forward_steps.jsonl",
                layer=args.layer,
                skip_initial_steps=args.skip_initial_steps,
                vmax_percentile=args.vmax_percentile,
                case_name=case_name,
                model_name=args.model_name,
                enable_eplb=("1" if case_name.endswith("_eplb") and "noeplb" not in case_name else "0"),
                count_divisor=count_divisor,
            )
        return

    if args.out_dir is None:
        parser.error("--out-dir is required when using --pt or --manifest")

    record_path = _resolve_record_path(args.pt, args.manifest)
    case_dir = _infer_case_dir_from_source(record_path, args.manifest)
    count_divisor = args.count_divisor or (
        _infer_count_divisor(case_dir) if case_dir is not None else 1
    )
    _process_record(
        record_path=record_path,
        manifest_path=args.manifest,
        out_dir=Path(args.out_dir).expanduser().resolve(),
        mapping_jsonl_path=(
            Path(args.mapping_jsonl).expanduser().resolve()
            if args.mapping_jsonl is not None
            else None
        ),
        record_meta_path=(
            Path(args.record_meta_json).expanduser().resolve()
            if args.record_meta_json is not None
            else None
        ),
        forward_steps_jsonl_path=(
            Path(args.forward_steps_jsonl).expanduser().resolve()
            if args.forward_steps_jsonl is not None
            else None
        ),
        layer=args.layer,
        skip_initial_steps=args.skip_initial_steps,
        vmax_percentile=args.vmax_percentile,
        case_name=args.case_name,
        model_name=args.model_name,
        enable_eplb=args.enable_eplb,
        count_divisor=count_divisor,
    )


if __name__ == "__main__":
    main()

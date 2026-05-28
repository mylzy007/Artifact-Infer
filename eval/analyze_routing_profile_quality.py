"""Analyze MoE routing profile row quality without touching runtime state."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.append(str(REPO))

from workshop.nanovllm_moe.services.utils.routing_profile_quality import compute_profile_quality


def load_profile(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def build_report(profile: dict[str, Any], profile_path: Path) -> dict[str, Any]:
    traffic = profile.get("traffic")
    if not isinstance(traffic, list):
        raise ValueError(f"profile {profile_path} has no valid 'traffic' field")

    quality = compute_profile_quality(traffic)
    per_layer = quality["per_layer"]
    summary = quality["summary"]
    return {
        "profile_path": str(profile_path.resolve()),
        "profile_run_id": profile.get("profile_run_id"),
        "world_size": profile.get("world_size"),
        "tp_size": profile.get("tp_size"),
        "ep_size": profile.get("ep_size"),
        "num_layers": profile.get("num_layers", len(per_layer)),
        "num_experts": profile.get("num_experts"),
        "prefill_decode_split": profile.get("prefill_decode_split", "combined"),
        "row_identical_fraction_per_layer": [
            layer["row_identical_fraction"] for layer in per_layer
        ],
        "row_pairwise_l1_distance_per_layer": [
            layer["row_pairwise_l1_distance"] for layer in per_layer
        ],
        "row_pairwise_cosine_similarity_per_layer": [
            layer["row_pairwise_cosine_similarity"] for layer in per_layer
        ],
        "row_entropy_per_src_per_layer": [
            layer["row_entropy_per_src"] for layer in per_layer
        ],
        "layer_all_rows_identical": [
            bool(layer["all_rows_identical"]) for layer in per_layer
        ],
        "summary": {
            "profile_identical_fraction_mean": summary["row_identical_fraction_mean"],
            "profile_pairwise_l1_mean": summary["row_pairwise_l1_distance_mean"],
            "profile_pairwise_cosine_mean": summary["row_pairwise_cosine_similarity_mean"],
            "profile_all_layers_identical": summary["all_layers_identical"],
            "fully_degenerate_layer_ids": [
                int(layer["layer_id"]) for layer in per_layer if layer["all_rows_identical"]
            ],
        },
        "per_layer": per_layer,
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Routing Profile Quality Report",
        "",
        f"- Profile: `{report['profile_path']}`",
        f"- TP={report.get('tp_size')} EP={report.get('ep_size')} world_size={report.get('world_size')}",
        f"- Layers={report.get('num_layers')} experts={report.get('num_experts')}",
        "",
        "## Summary",
        "",
        f"- identical_fraction_mean: `{report['summary']['profile_identical_fraction_mean']}`",
        f"- pairwise_l1_mean: `{report['summary']['profile_pairwise_l1_mean']}`",
        f"- pairwise_cosine_mean: `{report['summary']['profile_pairwise_cosine_mean']}`",
        f"- all_layers_identical: `{report['summary']['profile_all_layers_identical']}`",
        f"- fully_degenerate_layer_ids: `{report['summary']['fully_degenerate_layer_ids']}`",
        "",
        "## Per Layer",
        "",
        "| layer | identical_fraction | pairwise_l1_mean | pairwise_cosine_mean | entropy_mean | all_rows_identical |",
        "| ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for layer in report["per_layer"]:
        lines.append(
            "| {layer_id} | {identical} | {l1} | {cosine} | {entropy} | {identical_flag} |".format(
                layer_id=layer["layer_id"],
                identical=layer["row_identical_fraction"],
                l1=layer["row_pairwise_l1_distance"]["mean"],
                cosine=layer["row_pairwise_cosine_similarity"]["mean"],
                entropy=layer["row_entropy_summary"]["mean"],
                identical_flag=layer["all_rows_identical"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def write_optional(path: Path | None, content: str) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument(
        "--print-format",
        choices=["json", "markdown", "summary"],
        default="summary",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(load_profile(args.profile), args.profile)
    write_optional(args.output_json, json.dumps(report, indent=2))
    write_optional(args.output_md, markdown_report(report))

    if args.print_format == "json":
        print(json.dumps(report, indent=2))
        return
    if args.print_format == "markdown":
        print(markdown_report(report))
        return

    summary = report["summary"]
    print(
        json.dumps(
            {
                "profile_path": report["profile_path"],
                "tp_size": report.get("tp_size"),
                "ep_size": report.get("ep_size"),
                "profile_identical_fraction_mean": summary["profile_identical_fraction_mean"],
                "profile_pairwise_l1_mean": summary["profile_pairwise_l1_mean"],
                "profile_pairwise_cosine_mean": summary["profile_pairwise_cosine_mean"],
                "profile_all_layers_identical": summary["profile_all_layers_identical"],
                "fully_degenerate_layer_ids": summary["fully_degenerate_layer_ids"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

"""Re-score saved generations using official LongBench + ROUGE metrics.

Uses:
  - LongBench-style QA F1 (with article/punct removal + whitespace normalization),
    matches `THUDM/LongBench/metrics.py::qa_f1_score`.
  - Google's `rouge_score` package for ROUGE-1 / ROUGE-2 / ROUGE-L (with Porter stemmer),
    matches the canonical rouge_score package used by LEval / LongBench-v2.
  - LongBench-style classification accuracy (for completeness — not used on QA tasks).

For each plan dir under --root, reads {plan_dir}/v2_rows.jsonl, recomputes
per-cell metrics on the saved `gen_text` + `ref_text` fields, and writes
{root}/official_rescore.json + official_rescore_table.md.

Usage:
    python -m eval.drop.rescore_official \\
        --root eval_results/owner_local_ep_phase4_drop_longbench_v4
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import string
import sys
from collections import Counter, defaultdict

from rouge_score import rouge_scorer


# ----------------------------------------------------------------------
# LongBench official normalization (copied from THUDM/LongBench metrics.py)
# ----------------------------------------------------------------------

def _remove_articles(text: str) -> str:
    return re.sub(r"\b(a|an|the)\b", " ", text)


def _white_space_fix(text: str) -> str:
    return " ".join(text.split())


_PUNCT = set(string.punctuation)


def _remove_punc(text: str) -> str:
    return "".join(ch for ch in text if ch not in _PUNCT)


def normalize_answer(s: str) -> str:
    """LongBench's standard normalization for QA answers."""
    return _white_space_fix(_remove_articles(_remove_punc(s.lower())))


def longbench_qa_f1(prediction: str, ground_truth: str) -> float:
    """Match `THUDM/LongBench/metrics.py::qa_f1_score`."""
    norm_pred = normalize_answer(prediction).split()
    norm_gt = normalize_answer(ground_truth).split()
    common = Counter(norm_pred) & Counter(norm_gt)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(norm_pred) if norm_pred else 0.0
    recall = num_same / len(norm_gt) if norm_gt else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def longbench_qa_recall(prediction: str, ground_truth: str) -> float:
    """Same as official F1 but report only recall — robust to long verbose generations."""
    norm_pred = normalize_answer(prediction).split()
    norm_gt = normalize_answer(ground_truth).split()
    common = Counter(norm_pred) & Counter(norm_gt)
    num_same = sum(common.values())
    if num_same == 0 or not norm_gt:
        return 0.0
    return num_same / len(norm_gt)


def longbench_qa_substring(prediction: str, ground_truth: str) -> float:
    """1.0 iff every reference token appears somewhere in the prediction
    (using LongBench normalization). Forgiving for verbose generations."""
    norm_pred = set(normalize_answer(prediction).split())
    norm_gt = set(normalize_answer(ground_truth).split())
    if not norm_gt:
        return 0.0
    return 1.0 if norm_gt.issubset(norm_pred) else 0.0


# Google rouge_score (canonical, with Porter stemmer + sentence tokenization)
_rouge_scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def rouge_metrics(prediction: str, ground_truth: str) -> dict[str, float]:
    if not prediction or not ground_truth:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    scores = _rouge_scorer.score(ground_truth, prediction)  # arg order: target, prediction
    return {
        "rouge1": float(scores["rouge1"].fmeasure),
        "rouge2": float(scores["rouge2"].fmeasure),
        "rougeL": float(scores["rougeL"].fmeasure),
    }


# ----------------------------------------------------------------------
# Main rescore
# ----------------------------------------------------------------------

def rescore_dir(plan_dir: str) -> dict:
    """Re-score one plan directory; return {label: {metric: {mean, sem, n}}}."""
    rows_path = os.path.join(plan_dir, "v2_rows.jsonl")
    if not os.path.exists(rows_path):
        return {}
    cell_scores = defaultdict(lambda: defaultdict(list))
    for line in open(rows_path):
        line = line.strip()
        if not line: continue
        d = json.loads(line)
        label = d["label"]
        for g in d.get("generations", []):
            # v4 stores gen_text + ref_text; v3 only stored gen_text_head[:200]
            pred = g.get("gen_text") or g.get("gen_text_head", "")
            ref = g.get("ref_text", "")
            if not ref or not pred:
                continue
            cell_scores[label]["lb_f1"].append(longbench_qa_f1(pred, ref))
            cell_scores[label]["lb_recall"].append(longbench_qa_recall(pred, ref))
            cell_scores[label]["lb_substring"].append(longbench_qa_substring(pred, ref))
            r = rouge_metrics(pred, ref)
            cell_scores[label]["rouge1"].append(r["rouge1"])
            cell_scores[label]["rouge2"].append(r["rouge2"])
            cell_scores[label]["rougeL"].append(r["rougeL"])

    out = {}
    for label, metric_vals in cell_scores.items():
        out[label] = {}
        for mname, vals in metric_vals.items():
            out[label][mname] = {
                "mean": statistics.mean(vals) if vals else None,
                "sem": (statistics.stdev(vals) / (len(vals) ** 0.5))
                       if len(vals) > 1 else 0.0,
                "n": len(vals),
            }
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True,
                   help="Directory containing per-plan subdirs (each with v2_rows.jsonl)")
    args = p.parse_args()

    plans = {}
    for name in sorted(os.listdir(args.root)):
        plan_dir = os.path.join(args.root, name)
        if not os.path.isdir(plan_dir):
            continue
        if os.path.exists(os.path.join(plan_dir, "v2_rows.jsonl")):
            print(f"[rescore] processing {name}...")
            plans[name] = rescore_dir(plan_dir)

    with open(os.path.join(args.root, "official_rescore.json"), "w") as f:
        json.dump(plans, f, indent=2)

    # Markdown table
    lines = [
        "# Official re-score (LongBench-style F1 + rouge_score package)\n",
        "Metrics:",
        "- `lb_f1`: LongBench-official QA F1 (article/punct removal + token F1)",
        "- `lb_recall`: same normalization, recall-only (forgiving to long CoT)",
        "- `lb_substring`: 1 if every ref token appears in gen (normalized)",
        "- `rougeL`: Google rouge_score package with Porter stemmer\n",
        "| plan | label | n | lb_f1 | lb_recall | lb_substring | rouge1 | rouge2 | rougeL |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for plan_name in sorted(plans):
        for label in sorted(plans[plan_name]):
            c = plans[plan_name][label]
            def fmt(k):
                v = c.get(k, {}).get("mean")
                return f"{v:.4f}" if v is not None else "—"
            n = c.get("lb_f1", {}).get("n", 0)
            lines.append(f"| {plan_name} | {label} | {n} | "
                         f"{fmt('lb_f1')} | {fmt('lb_recall')} | {fmt('lb_substring')} | "
                         f"{fmt('rouge1')} | {fmt('rouge2')} | {fmt('rougeL')} |")
    open(os.path.join(args.root, "official_rescore_table.md"), "w").write("\n".join(lines))

    # Print baseline comparison
    print(f"\n=== Baseline metrics (across plans) ===")
    print(f"{'plan':<22}{'lb_f1':>9}{'lb_recall':>11}{'lb_substr':>11}{'rouge1':>9}{'rougeL':>9}")
    for plan_name in sorted(plans):
        if "baseline" not in plans[plan_name]: continue
        b = plans[plan_name]["baseline"]
        line = f"{plan_name:<22}"
        for m in ("lb_f1", "lb_recall", "lb_substring", "rouge1", "rougeL"):
            v = b.get(m, {}).get("mean")
            line += f"{v:>9.4f}" if v is not None else f"{'—':>9}"
            if m == "lb_recall": line = line.replace(f"{v:>9.4f}", f"{v:>11.4f}", 1) if v else line
            if m == "lb_substring": line = line.replace(f"{v:>9.4f}", f"{v:>11.4f}", 1) if v else line
        print(line)

    print(f"\n[rescore_official] wrote {args.root}/official_rescore.json + official_rescore_table.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())

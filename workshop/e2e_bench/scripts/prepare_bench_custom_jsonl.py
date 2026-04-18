#!/usr/bin/env python3
"""Prepare local benchmark data into vLLM custom JSONL format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable
import zipfile

try:
    import pandas as pd
except ImportError:
    pd = None


def pick_source_files(dataset: str, source: Path, subset: str | None) -> list[Path | str]:
    if source.is_file():
        if dataset == "longbench" and source.suffix == ".zip":
            subset_name = subset or "triviaqa"
            return [f"zip::{source}::data/{subset_name}.jsonl"]
        return [source]

    files = sorted(source.rglob("*.parquet"))

    if dataset == "openorca":
        if not files:
            raise FileNotFoundError(f"No parquet files found under {source}")
        preferred = [
            source / "1M-GPT4-Augmented.parquet",
            source / "3_5M-GPT3_5-Augmented.parquet",
        ]
        selected = [p for p in preferred if p.exists()]
        return selected or files

    if dataset == "gsm8k":
        if not files:
            raise FileNotFoundError(f"No parquet files found under {source}")
        preferred = [
            source / "main" / "test-00000-of-00001.parquet",
            source / "main" / "train-00000-of-00001.parquet",
        ]
        selected = [p for p in preferred if p.exists()]
        return selected or files

    if dataset == "longbench":
        subset_name = subset or "triviaqa"
        jsonl_path = source / "extracted" / "data" / f"{subset_name}.jsonl"
        if jsonl_path.exists():
            return [jsonl_path]
        zip_path = source / "data.zip"
        if zip_path.exists():
            return [f"zip::{zip_path}::data/{subset_name}.jsonl"]
        raise FileNotFoundError(
            f"Neither {jsonl_path} nor {zip_path} exists for LongBench subset {subset_name}"
        )

    if dataset == "longbench_v2":
        json_path = source / "data.json"
        if json_path.exists():
            return [json_path]
        raise FileNotFoundError(f"{json_path} does not exist for LongBench-v2")

    if dataset == "leval":
        subset_name = (subset or "Generation/multidoc_qa").strip()
        if subset_name.endswith(".jsonl"):
            candidate_names = [subset_name]
        elif "/" in subset_name:
            candidate_names = [f"{subset_name}.jsonl"]
        else:
            candidate_names = [
                f"Generation/{subset_name}.jsonl",
                f"Exam/{subset_name}.jsonl",
            ]
        for candidate_name in candidate_names:
            candidate = source / "LEval" / candidate_name
            if candidate.exists():
                return [candidate]
        raise FileNotFoundError(
            f"Could not find LEval subset {subset_name!r} under {source / 'LEval'}"
        )

    if not files:
        raise FileNotFoundError(f"No parquet files found under {source}")
    return files


def iter_records(source_files: Iterable[Path | str]) -> Iterable[dict]:
    for source_file in source_files:
        if isinstance(source_file, str) and source_file.startswith("zip::"):
            _, zip_path_str, member = source_file.split("::", 2)
            with zipfile.ZipFile(zip_path_str) as zf:
                with zf.open(member) as f:
                    for line in f:
                        if line.strip():
                            yield json.loads(line)
            continue

        path = Path(source_file)
        if path.suffix == ".parquet":
            if pd is None:
                raise ModuleNotFoundError(
                    "pandas is required to read parquet benchmark sources."
                )
            df = pd.read_parquet(path)
            for row in df.to_dict(orient="records"):
                yield row
            continue

        if path.suffix == ".json":
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for row in data:
                    if isinstance(row, dict):
                        yield row
                continue
            if isinstance(data, dict):
                yield data
                continue
            raise TypeError(f"Unsupported JSON payload type in {path}: {type(data).__name__}")

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def normalize_openorca(row: dict) -> list[tuple[str, str]]:
    question = str(row.get("question", "") or "").strip()
    response = str(row.get("response", "") or row.get("answer", "") or "").strip()
    system_prompt = str(row.get("system_prompt", "") or "").strip()
    if not question or not response:
        return []

    prompt_parts = []
    if system_prompt:
        prompt_parts.append(system_prompt)
    prompt_parts.append(question)
    return [("\n\n".join(prompt_parts), response)]


def normalize_gsm8k(row: dict) -> list[tuple[str, str]]:
    question = str(row.get("question", "") or "").strip()
    answer = str(row.get("answer", "") or "").strip()
    if not question or not answer:
        return []
    return [(question, answer)]


def normalize_longbench(row: dict) -> list[tuple[str, str]]:
    context = str(row.get("context", "") or "").strip()
    question = str(row.get("input", "") or "").strip()
    answers = row.get("answers") or []
    answer = str(answers[0]).strip() if answers else ""
    if not answer:
        return []
    prompt_parts = [part for part in (context, question) if part]
    prompt = "\n\n".join(prompt_parts).strip()
    if not prompt:
        return []
    return [(prompt, answer)]


def normalize_longbench_v2(row: dict) -> list[tuple[str, str]]:
    context = str(row.get("context", "") or "").strip()
    question = str(row.get("question", "") or "").strip()
    answer_key = str(row.get("answer", "") or "").strip()
    if not question or not answer_key:
        return []

    choice_map = {
        "A": str(row.get("choice_A", "") or "").strip(),
        "B": str(row.get("choice_B", "") or "").strip(),
        "C": str(row.get("choice_C", "") or "").strip(),
        "D": str(row.get("choice_D", "") or "").strip(),
    }
    choices = [f"{label}. {text}" for label, text in choice_map.items() if text]
    prompt_parts = []
    if context:
        prompt_parts.append(context)
    prompt_parts.append(question)
    if choices:
        prompt_parts.append("\n".join(choices))
    prompt = "\n\n".join(part for part in prompt_parts if part)

    response = choice_map.get(answer_key, "") or answer_key
    if not response:
        return []
    return [(prompt, response)]


def normalize_leval(row: dict) -> list[tuple[str, str]]:
    context = str(row.get("input", "") or "").strip()
    instructions = row.get("instructions") or []
    outputs = row.get("outputs") or []
    if isinstance(instructions, str):
        instructions = [instructions]
    if isinstance(outputs, str):
        outputs = [outputs]

    pairs: list[tuple[str, str]] = []
    for instruction, output in zip(instructions, outputs):
        question = str(instruction or "").strip()
        answer = str(output or "").strip()
        if not question or not answer:
            continue
        prompt = f"{context}\n\nQuestion:\n{question}".strip() if context else question
        pairs.append((prompt, answer))
    return pairs


NORMALIZERS = {
    "openorca": normalize_openorca,
    "gsm8k": normalize_gsm8k,
    "longbench": normalize_longbench,
    "longbench_v2": normalize_longbench_v2,
    "leval": normalize_leval,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(NORMALIZERS.keys()), required=True)
    parser.add_argument("--source", required=True, help="Source parquet file or directory")
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument(
        "--output-tokens",
        type=int,
        default=32,
        help="Fixed expected output length to embed into each vLLM custom row.",
    )
    parser.add_argument(
        "--subset",
        default=None,
        help="Optional subset/task name for datasets that expose multiple source files, e.g. LongBench.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the output even if it already exists",
    )
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()

    if out.exists() and out.stat().st_size > 0 and not args.force:
        print(f"[skip] prepared dataset already exists: {out}")
        return 0

    source_files = pick_source_files(args.dataset, source, args.subset)
    out.parent.mkdir(parents=True, exist_ok=True)

    normalizer = NORMALIZERS[args.dataset]
    subset_label = (args.subset or "").strip().replace("/", "__").replace(" ", "_")
    request_id_prefix = args.dataset if not subset_label else f"{args.dataset}.{subset_label}"
    num_written = 0
    with out.open("w", encoding="utf-8") as f:
        for row in iter_records(source_files):
            normalized_pairs = normalizer(row)
            if not normalized_pairs:
                continue
            for prompt, response in normalized_pairs:
                obj = {
                    "request_id": f"{request_id_prefix}.{num_written:06d}",
                    "prompt": prompt,
                    "output_tokens": int(args.output_tokens),
                    "reference_answer": response,
                }
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                num_written += 1

    if num_written == 0:
        raise RuntimeError(f"No usable records were written to {out}")

    print(f"[done] wrote {num_written} rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

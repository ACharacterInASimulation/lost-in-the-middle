#!/usr/bin/env python3
"""Summarize scored QA and KV prediction files."""
import argparse
import glob
import json
import pathlib
import statistics

from common_io import xopen


def main(input_paths, output_format):
    paths = expand_paths(input_paths)
    rows = []
    for path in paths:
        examples = read_examples(path)
        if not examples:
            continue
        rows.append(summarize_file(path, examples))

    if output_format == "tsv":
        print_tsv(rows)
    else:
        print_markdown(rows)


def expand_paths(input_paths):
    paths = []
    for input_path in input_paths:
        expanded = sorted(glob.glob(input_path, recursive=True))
        paths.extend(expanded if expanded else [input_path])
    return paths


def read_examples(path):
    examples = []
    with xopen(path) as fin:
        for line in fin:
            examples.append(json.loads(line))
    return examples


def summarize_file(path, examples):
    first = examples[0]
    metric_names = sorted(k for k in first if k.startswith("metric_"))
    metrics = {}
    for metric_name in metric_names:
        values = [example[metric_name] for example in examples if metric_name in example]
        if values:
            metrics[metric_name.removeprefix("metric_")] = statistics.mean(values)

    task = infer_task(first)
    total_items, gold_index = infer_position(first, task)
    return {
        "path": path,
        "task": task,
        "model": first.get("model", ""),
        "total_items": total_items,
        "gold_index": gold_index,
        "num_examples": len(examples),
        "metrics": metrics,
    }


def infer_task(example):
    if "model_ordered_kv_records" in example or "ordered_kv_records" in example:
        return "kv"
    if "model_documents" in example or "ctxs" in example:
        return "qa"
    return "unknown"


def infer_position(example, task):
    if task == "kv":
        records = example.get("model_ordered_kv_records") or example.get("ordered_kv_records") or []
        gold_pair = [example.get("key"), example.get("value")]
        if gold_pair in records:
            return len(records), records.index(gold_pair)
        return len(records), example.get("model_gold_index", "")

    if task == "qa":
        documents = example.get("model_documents") or example.get("ctxs") or []
        for index, document in enumerate(documents):
            if document.get("isgold") is True:
                return len(documents), index
        return len(documents), ""

    return "", ""


def print_tsv(rows):
    metric_names = sorted({metric_name for row in rows for metric_name in row["metrics"]})
    headers = ["task", "model", "total_items", "gold_index", "num_examples"] + metric_names + ["path"]
    print("\t".join(headers))
    for row in rows:
        values = [
            row["task"],
            row["model"],
            str(row["total_items"]),
            str(row["gold_index"]),
            str(row["num_examples"]),
        ]
        values.extend(format_metric(row["metrics"].get(metric_name)) for metric_name in metric_names)
        values.append(row["path"])
        print("\t".join(values))


def print_markdown(rows):
    metric_names = sorted({metric_name for row in rows for metric_name in row["metrics"]})
    headers = ["task", "model", "items", "gold", "n"] + metric_names + ["path"]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        values = [
            row["task"],
            row["model"],
            str(row["total_items"]),
            str(row["gold_index"]),
            str(row["num_examples"]),
        ]
        values.extend(format_metric(row["metrics"].get(metric_name)) for metric_name in metric_names)
        values.append(pathlib.Path(row["path"]).as_posix())
        print("| " + " | ".join(values) + " |")


def format_metric(value):
    if value is None:
        return ""
    return f"{value:.6f}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_paths", nargs="+", help="Scored prediction JSONL(.gz) files or shell globs")
    parser.add_argument("--format", choices=["markdown", "tsv"], default="markdown")
    args = parser.parse_args()
    main(args.input_paths, args.format)

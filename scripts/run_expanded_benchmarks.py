"""Evaluate frozen quantization recipes on explicitly pinned added datasets.

Upstream test records are intentionally evaluated, so this driver marks that
fact explicitly. Existing calibration and development files are reused without
using any new evaluation target for fitting or configuration selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from s1q.data import checked_datasets
from s1q.optimization import add_arguments, run

METHODS = "rtn,s1q-local,s1q2-beta05,awq-adapted,gptq-blockdiag-adapted,spinquant-nohad-adapted,spinquant-had-adapted,s1q-joint,s1q-ac,s1q-margin,s1q"


def main():
    parser = add_arguments(argparse.ArgumentParser(description=__doc__), default_methods=METHODS)
    parser.add_argument("--benchmark-manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest, datasets = checked_datasets(args.benchmark_manifest)
    if args.extra_evaluation:
        raise ValueError("Extra datasets must be recorded in the pinned benchmark manifest")
    args.extra_evaluation = [f"{name}={path}" for name, path in datasets]
    args.cohort = "frozen_expanded_evaluation_20261007"
    # Stored in the before-evaluation configuration, rather than inferred later.
    args.benchmark_manifest_sha256 = hashlib.sha256(args.benchmark_manifest.read_bytes()).hexdigest()
    args.benchmark_manifest = str(args.benchmark_manifest)
    args.upstream_test_records_intentionally_evaluated = True
    args.recipe_selected_before_expanded_evaluation = True
    report = run(args)
    identity = report["identity"]
    identity["final_test_read"] = True
    identity["evaluation_role"] = "frozen request-disjoint expanded evaluation; upstream test/development provenance in manifest"
    identity["benchmark_manifest_sha256"] = args.benchmark_manifest_sha256
    identity["expanded_ranking_dataset_ids"] = [name for name, _ in datasets]
    identity["old_development_excluded_from_expanded_ranking"] = True
    identity["benchmark_id"] = manifest.get("benchmark_id", manifest.get("stage_id"))
    for name, records in identity["accepted_evaluation_ids"].items():
        if name != "development" and not records:
            raise ValueError(f"Entire dataset rejected by native model: {name}")
    output = Path(args.output_dir)
    for filename, value in (("report.json", report), ("identity.json", identity)):
        (output / filename).write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
    (output / "expanded-evaluation-complete.json").write_text(json.dumps({
        "benchmark_manifest_sha256": args.benchmark_manifest_sha256,
        "ranking_dataset_ids": identity["expanded_ranking_dataset_ids"],
        "status": "complete" if all(x["status"] == "complete" for x in report["methods"].values()) else "method_failure",
        "methods": {k: v["status"] for k, v in report["methods"].items()},
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

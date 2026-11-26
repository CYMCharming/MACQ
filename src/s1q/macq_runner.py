"""Command line entry point for the published MACQ benchmark profile."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import random
from statistics import mean

import numpy as np
import torch

from .api import CALIBRATION_SEED, quantize_macq
from .data import assert_disjoint, checked_datasets, load_records
from .experiment import predict, write_rows
from .margin_reconstruction import _file_hash, _json
from .models import load_model


def evaluate_benchmark(adapter, calibration, manifest_path, out):
    manifest, datasets = checked_datasets(manifest_path)
    counts = {entry["dataset_id"]: entry["n_decisions"] for entry in manifest["datasets"]}
    cohorts = {name: load_records(path) for name, path in datasets}
    assert_disjoint({"calibration": calibration, **cohorts})
    predictions = out / "predictions"
    predictions.mkdir()
    scores = {}
    for name, records in cohorts.items():
        rows, _, _, _ = predict(adapter, records)
        if len(rows) != counts[name]:
            raise ValueError(f"Benchmark decision count differs: {name}")
        correct = sum(
            max(range(len(row["logits"])), key=row["logits"].__getitem__) == row["label"]
            for row in rows
        )
        scores[name] = {
            "decisions": len(rows),
            "correct": correct,
            "accuracy_percent": 100 * correct / len(rows),
        }
        write_rows(predictions / f"{name}.jsonl", rows)
    report = {
        "benchmark_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "suites": len(scores),
        "decisions": sum(entry["decisions"] for entry in scores.values()),
        "macro_accuracy_percent": mean(entry["accuracy_percent"] for entry in scores.values()),
        "per_suite": scores,
    }
    _json(out / "scores.json", report)


def run(args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(CALIBRATION_SEED)
    np.random.seed(CALIBRATION_SEED)
    torch.manual_seed(CALIBRATION_SEED)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    path = args.data_dir / "calibration.jsonl"
    calibration = load_records(path)
    adapter = load_model(
        args.model,
        device=args.device,
        dtype=args.dtype,
        source_dir=args.source_dir,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_manifest=args.checkpoint_manifest,
    )
    adapter.model.eval()
    with quantize_macq(
        adapter,
        calibration,
        rotation_source=args.rotation_source,
        output_dir=args.output_dir,
        rotation_checkpoint=args.rotation_checkpoint,
    ) as (_, payload):
        rotation_path = args.output_dir / "trained-rotations.pt"
        _json(
            args.output_dir / "run.json",
            {
                "method": "MACQ",
                "model": adapter.metadata,
                "torch": str(torch.__version__),
                "cuda": torch.version.cuda,
                "weight_bits": 4,
                "activation_bits": 4,
                "calibration_sha256": _file_hash(path),
                "use_supplied_rotation": args.rotation_checkpoint is not None,
                "trained_rotation_file_sha256": _file_hash(rotation_path),
                "configuration": payload["config"],
            },
        )
        if args.benchmark_manifest is not None:
            evaluate_benchmark(adapter, calibration, args.benchmark_manifest, args.output_dir)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    optimize = commands.add_parser("optimize", help="Fit and optionally evaluate MACQ W4A4")
    optimize.add_argument("--model", required=True)
    optimize.add_argument("--data-dir", type=Path, required=True)
    optimize.add_argument("--output-dir", type=Path, required=True)
    optimize.add_argument("--source-dir", type=Path, default=Path("work/upstream"))
    optimize.add_argument("--rotation-source", type=Path, default=Path("work/upstream/rotation"))
    optimize.add_argument("--rotation-checkpoint", type=Path)
    optimize.add_argument("--checkpoint-dir", type=Path)
    optimize.add_argument("--checkpoint-manifest", type=Path)
    optimize.add_argument("--benchmark-manifest", type=Path)
    optimize.add_argument("--device", default="cuda")
    optimize.add_argument("--dtype", default="bf16")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

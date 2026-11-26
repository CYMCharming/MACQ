"""Public calibration-and-quantization entry points for MACQ."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from shutil import copyfile
from typing import Iterable

import torch

from .data import model_request
from .decision_margin import DecisionMarginCollector
from .margin_reconstruction import activate_payloads, collect_margin_pools, fit_payloads
from .optimization import subset
from .optimized_quantization import quantize_optimized_model
from .quantization import selected_linear_modules
from .rotation import RotationSession


CALIBRATION_SEED = 20261004
CALIBRATION_REQUESTS = 128


def benchmark_cap(model):
    """Frozen per-model compensation settings used in the public benchmark."""
    return {
        "startlux-decision-0.8b": 0.05,
        "startlux-decision-2b": None,
        "startlux-decision-4b": None,
    }.get(model, 0.10)


def quantize_s1q(adapter, calibration_records: Iterable[dict], *, bits=4,
                 activation_bits=4, group_size=128, reservoir_size=128,
                 seed=20261004):
    """Apply S1Q to an evaluation-mode native adapter and return its session.

    Only native-admitted calibration requests belong here. Gold labels and
    targets are stripped from independent copies before sensitivity collection.
    The returned session is already active; use it as a context manager or call
    ``restore()``. Calibration performs backward passes for token importance;
    inference remains floating-point with weight/activation QDQ.
    """
    if adapter.model.training:
        raise ValueError("Put the native model in eval mode before S1Q calibration")
    count = 0
    with DecisionMarginCollector(adapter, modules=selected_linear_modules(adapter.backbone),
                                 reservoir_size=reservoir_size, seed=seed) as collector:
        for record in calibration_records:
            request = deepcopy(record)
            for question in request["questions"].values():
                question.pop("label", None)
                question.pop("target", None)
            collector.collect(request)
            count += 1
    if count == 0:
        raise ValueError("S1Q calibration requires at least one admitted request")
    session, layer_metadata = quantize_optimized_model(
        adapter.backbone, method="s1q", statistics=collector.statistics(),
        bits=bits, group_size=group_size, activation_bits=activation_bits,
        compensation=True)
    return session, {"method": "S1Q", "calibration": collector.metadata(),
                     "layers": layer_metadata,
                     "activation_execution": "floating QDQ; no integer kernel"}


def _load_rotation(session, path):
    session.load_rotations(torch.load(path, map_location="cpu", weights_only=True))
    _freeze_rotation(session)


def _freeze_rotation(session):
    if max(session.orthogonality().values()) > 1e-3:
        raise ValueError("Rotation orthogonality exceeds the benchmark tolerance")
    for parameter in session.parameters():
        parameter.requires_grad_(False)


@contextmanager
def quantize_macq(adapter, calibration_records: Iterable[dict], *, rotation_source,
                  output_dir, rotation_checkpoint=None):
    """Fit and activate the published W4A4 profile on a native adapter.

    The output directory receives reusable packed weights and the learned
    rotations. Optional pre-trained rotations must match the adapter's native
    topology. Only the 128 admitted calibration requests reach the fitter.
    """
    if adapter.model.training or torch.is_inference_mode_enabled():
        raise ValueError("Calibrate an eval-mode adapter outside torch.inference_mode()")
    records = list(calibration_records)
    if len(records) != CALIBRATION_REQUESTS:
        raise ValueError(f"MACQ requires {CALIBRATION_REQUESTS} native calibration requests")
    requests = [model_request(record) for record in subset(records, CALIBRATION_REQUESTS, CALIBRATION_SEED)]
    out = Path(output_dir)
    out.mkdir(parents=True)
    checkpoint = out / "trained-rotations.pt"
    session = RotationSession(
        adapter.backbone, rotation_source, online_had=True, seed=0,
        protected_model=adapter.model, checkpoint_blocks=rotation_checkpoint is None,
    )
    try:
        if rotation_checkpoint is None:
            with torch.enable_grad():
                session.fit(requests, adapter.infer, out=out)
            _freeze_rotation(session)
            torch.save(session.rotation_state(), checkpoint)
        else:
            copyfile(rotation_checkpoint, checkpoint)
            _load_rotation(session, checkpoint)
        pools = collect_margin_pools(adapter, session, requests, out)
        fit_payloads(session, pools, out, correction_cap=benchmark_cap(adapter.name))
        metadata = activate_payloads(session, out)
        yield session, metadata
    finally:
        session.close()


@contextmanager
def load_macq(adapter, output_dir, *, rotation_source):
    """Activate existing MACQ packed weights on the same native model."""
    if adapter.model.training:
        raise ValueError("Put the native adapter in eval mode before loading MACQ")
    out = Path(output_dir)
    session = RotationSession(adapter.backbone, rotation_source, online_had=True, seed=0,
                              protected_model=adapter.model, checkpoint_blocks=False)
    try:
        _load_rotation(session, out / "trained-rotations.pt")
        metadata = activate_payloads(session, out)
        yield session, metadata
    finally:
        session.close()

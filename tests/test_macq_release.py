import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from s1q import api, macq_runner
from s1q.margin_reconstruction import fit_payloads
from s1q.rotation import RotationSession


class TinyDecision:
    name = "startlux-decision-0.8b"

    def __init__(self):
        self.model = self.backbone = nn.Module()
        self.model.config = SimpleNamespace(hidden_size=8, use_cache=True)
        block = nn.Module()
        block.mlp = nn.ModuleDict({"down_proj": nn.Linear(8, 8)})
        block.linear_attn = nn.Module()
        block.linear_attn.head_v_dim = 4
        block.linear_attn.out_proj = nn.Linear(8, 8)
        self.model.layers = nn.ModuleList([block])
        self.model.eval()

    def infer(self, request):
        assert "label" not in request["questions"]["q"]
        block = self.backbone.layers[0]
        x = block.linear_attn.out_proj(torch.arange(8).float().unsqueeze(0))
        return [block.mlp["down_proj"](x)[0, :2]]


def test_published_cli_and_profiles(monkeypatch):
    captured = []
    monkeypatch.setattr(macq_runner, "run", lambda args: captured.append(args) or 0)
    assert (
        macq_runner.main(
            [
                "optimize",
                "--model",
                "kev-4b",
                "--data-dir",
                "data",
                "--output-dir",
                "out",
                "--benchmark-manifest",
                "manifest.json",
            ]
        )
        == 0
    )
    (args,) = captured
    assert args.rotation_source == Path("work/upstream/rotation")
    assert args.benchmark_manifest == Path("manifest.json")
    assert api.benchmark_cap("kev-4b") == 0.10
    assert api.benchmark_cap("startlux-decision-0.8b") == 0.05
    assert api.benchmark_cap("startlux-decision-2b") is None
    assert api.benchmark_cap("startlux-decision-4b") is None


def test_packed_w4a4_reload_and_native_restoration(tmp_path, monkeypatch):
    backend = Path(os.environ.get("MACQ_TEST_ROTATION_SOURCE", "work/upstream/rotation"))
    if not backend.is_dir():
        pytest.skip("Run scripts/setup_sources.py --families rotation to check the pinned backend")
    torch.manual_seed(203)
    adapter = TinyDecision()
    linear = adapter.model.layers[0].mlp["down_proj"]
    original = linear.weight.detach().clone()
    rotation = tmp_path / "input-rotations.pt"
    session = RotationSession(adapter.backbone, backend, online_had=True, checkpoint_blocks=False)
    try:
        torch.save(session.rotation_state(), rotation)
    finally:
        session.close()

    question = {"type": "noul", "instructions": "Is this resolved?", "label": 1}
    records = [
        {"id": str(i), "state": str(i), "questions": {"q": dict(question)}} for i in range(128)
    ]
    clean = {
        "state": "sample",
        "questions": {"q": {"type": "noul", "instructions": "Is this resolved?"}},
    }
    unquantized = adapter.infer(clean)[0].detach().clone()

    def calibration(adapter, session, requests, out):
        assert len(requests) == 128
        with session.mode(enabled=False):
            for request in requests:
                adapter.infer(request)
        return {name: torch.randn(32, 8) for name in session.wrappers}

    def fit_tiny(session, pools, out, *, correction_cap):
        assert correction_cap == 0.05
        return fit_payloads(
            session,
            pools,
            out,
            correction_cap=correction_cap,
            config={
                "technical_test_only": True,
                "pool_size": 32,
                "fit_rows": 16,
                "select_rows": 16,
            },
        )

    monkeypatch.setattr(api, "collect_margin_pools", calibration)
    monkeypatch.setattr(api, "fit_payloads", fit_tiny)
    out = tmp_path / "packed"
    with api.quantize_macq(
        adapter, records, rotation_source=backend, output_dir=out, rotation_checkpoint=rotation
    ) as (_, payload), torch.inference_mode():
        result = adapter.infer(clean)[0]
        assert torch.isfinite(result).all()
        assert payload["config"]["correction_cap"] == 0.05
        assert payload["config"]["weight_bits"] == payload["config"]["activation_bits"] == 4
        assert payload["layers"]["layers.0.mlp.down_proj"]["plan"]["online_had"]
    with api.load_macq(adapter, out, rotation_source=backend), torch.inference_mode():
        torch.testing.assert_close(adapter.infer(clean)[0], result, rtol=0, atol=0)
    assert records[0]["questions"]["q"]["label"] == 1
    assert adapter.model.layers[0].mlp["down_proj"] is linear
    assert linear.weight.requires_grad
    torch.testing.assert_close(linear.weight, original, rtol=0, atol=0)
    torch.testing.assert_close(adapter.infer(clean)[0], unquantized, rtol=0, atol=0)

    receipt = json.loads((out / "payload-manifest.json").read_text(encoding="utf-8"))
    file = (
        out / "payloads" / (receipt["layers"]["layers.0.mlp.down_proj"]["payload_sha256"] + ".pt")
    )
    with file.open("ab") as stream:
        stream.write(b"tampering")
    with pytest.raises(ValueError, match="payload file hash mismatch"), api.load_macq(
        adapter,
        out,
        rotation_source=backend,
    ):
        pytest.fail("Modified packed weights must not activate")
    assert adapter.model.layers[0].mlp["down_proj"] is linear
    assert linear.weight.requires_grad

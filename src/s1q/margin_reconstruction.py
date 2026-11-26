"""Decision-margin calibration and post-rotation W4A4 reconstruction."""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F


DEFAULT_CONFIG = {
    "pool_size": 16384,
    "fit_rows": 8192,
    "select_rows": 8192,
    "seed": 20261004,
    "ridge": 0.1,
    "correction_cap": 0.1,
    "alphas": [0.0, 0.25, 0.5, 0.75, 0.9, 1.0],
    "families": ["rms", "mean_abs"],
    "weight_bits": 4,
    "activation_bits": 4,
    "weight_groupsize": -1,
    "weight_perchannel": True,
    "weight_symmetric": True,
    "weight_mse": True,
    "weight_mse_grid": 100,
    "weight_mse_maxshrink": 0.8,
    "weight_mse_norm": 2.4,
    "activation_symmetric": False,
    "activation_clip_ratio": 1.0,
    "activation_groups": "native value-head dimension or full token",
    "external_weight_clipping": False,
    "candidate_count_bounded": 23,
    "candidate_count_no_comp": 12,
    "split_unit": "sampled native token row",
    "split_is_request_disjoint": False,
    "output_chunk": 512,
    "row_chunk": 1024,
    "weight_quantizer_row_chunk": 512,
}


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _json_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def tensor_hash(value):
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256(str((str(value.dtype), tuple(value.shape))).encode())
    # Stream along dimension zero; avoid a second whole-model-sized byte copy.
    rows = value.reshape(-1)
    for part in rows.split(1 << 20):
        digest.update(part.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for part in iter(lambda: handle.read(1 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def _rotation_hash(session):
    return _json_hash({k: tensor_hash(v) for k, v in session.rotation_state().items()})


def _configuration(config):
    result = dict(DEFAULT_CONFIG)
    if config:
        unknown = set(config) - set(result) - {"technical_test_only"}
        if unknown:
            raise ValueError("Unknown MAC configuration: " + str(sorted(unknown)))
        result.update(config)
    technical = bool(result.get("technical_test_only", False))
    for key in (
        "ridge",
        "correction_cap",
        "alphas",
        "families",
        "weight_bits",
        "activation_bits",
        "weight_groupsize",
        "weight_perchannel",
        "weight_symmetric",
        "weight_mse",
        "weight_mse_grid",
        "weight_mse_maxshrink",
        "weight_mse_norm",
        "activation_symmetric",
        "activation_clip_ratio",
        "external_weight_clipping",
    ):
        if result[key] != DEFAULT_CONFIG[key]:
            raise ValueError("Frozen numerical protocol cannot be changed: " + key)
    if not technical and any(
        result[k] != DEFAULT_CONFIG[k] for k in ("pool_size", "fit_rows", "select_rows", "seed")
    ):
        raise ValueError("Formal run requires 16K, 8192/8192, fixed seed")
    if result["fit_rows"] + result["select_rows"] != result["pool_size"]:
        raise ValueError("Fit/select must partition every sampled row")
    if (
        min(
            result["fit_rows"],
            result["select_rows"],
            result["output_chunk"],
            result["row_chunk"],
            result["weight_quantizer_row_chunk"],
        )
        < 1
    ):
        raise ValueError("Invalid row/chunk budget")
    return result


def rotate_inputs(wrapper, inputs):
    """Exactly the pre-A4 coordinate operations of frozen rotated Linear."""
    from .rotation import grouped_right, online_hadamard

    owner = wrapper.owner_ref()
    value = inputs.float()
    if wrapper.plan["input_r1"]:
        value = value @ owner.R1
    if wrapper.plan.get("r2_key"):
        value = grouped_right(value, owner.R2[wrapper.plan["r2_key"]])
    if wrapper.plan["online_had"]:
        value = online_hadamard(value, wrapper.plan, owner.official)
    return value


def _new_aq(official, plan):
    aq = official.ActQuantizer()
    aq.configure(4, groupsize=plan.get("head_dim", -1), sym=False, clip_ratio=1.0)
    return aq


@torch.no_grad()
def activation_qdq(official, plan, value, *, row_chunk=1024):
    aq = _new_aq(official, plan)
    parts = []
    for part in value.split(row_chunk):
        aq.find_params(part)
        parts.append(aq(part))
        aq.free()
    result = torch.cat(parts)
    if not torch.isfinite(result).all():
        raise ValueError("Official A4 produced a nonfinite value")
    return result


def pack_int4(codes):
    if codes.dtype != torch.int8 or codes.ndim != 2 or bool(((codes < -8) | (codes > 7)).any()):
        raise ValueError("Expected signed official W4 codes in [-8,7]")
    flat = (codes.contiguous().view(-1).to(torch.int16) & 15).to(torch.uint8)
    if flat.numel() % 2:
        flat = torch.cat((flat, torch.zeros(1, dtype=torch.uint8, device=flat.device)))
    return (flat[0::2] | (flat[1::2] << 4)).cpu()


def unpack_int4(packed, shape, *, device="cpu"):
    values = packed.to(device=device)
    unsigned = torch.stack((values & 15, values >> 4), dim=1).reshape(-1)
    signed = unsigned.to(torch.int16)
    signed = torch.where(signed >= 8, signed - 16, signed).to(torch.int8)
    return signed[: math.prod(shape)].reshape(shape)


def decode_payload(payload, *, device="cpu"):
    codes = unpack_int4(payload["packed_codes"], payload["shape"], device=device)
    return payload["row_scale"].to(device=device) * codes.float()


def _make_payload(codes, scales, input_scale, *, is_c0=False):
    if (
        scales.shape != (codes.shape[0], 1)
        or not torch.isfinite(scales).all()
        or bool((scales <= 0).any())
    ):
        raise ValueError("Official per-output W4 scales must be positive [out,1]")
    return {
        "format": "signed-int4-two-per-byte-low-nibble-first",
        "shape": list(codes.shape),
        "packed_codes": pack_int4(codes),
        "row_scale": scales.detach().float().cpu().clone(),
        "input_scale": input_scale.detach().float().cpu().clone(),
        "is_exact_c0": bool(is_c0),
        "floating_residual": False,
        "weight_bits": 4,
        "activation_bits": 4,
    }


@torch.no_grad()
def quantize_payload(official, weight, input_scale, *, row_chunk=512):
    """Unchanged official row-independent MSE quantizer; stream output rows."""
    codes, scales, qdq = [], [], []
    for part in weight.split(row_chunk):
        quantizer = official.WeightQuantizer()
        quantizer.configure(4, perchannel=True, sym=True, mse=True, weight_groupsize=-1)
        quantizer.find_params(part)
        reconstructed, integer, scale = quantizer.fake_quantize(part)
        expected = quantizer.quantize(part)
        if not torch.equal(reconstructed, expected):
            raise AssertionError("Official integer payload and QDQ disagree")
        codes.append(integer.to(torch.int8).cpu())
        scales.append(scale.cpu())
        qdq.append(reconstructed)
    payload = _make_payload(torch.cat(codes), torch.cat(scales), input_scale)
    result = torch.cat(qdq)
    if not torch.equal(decode_payload(payload, device=weight.device), result):
        raise AssertionError("INT4 packing changed official quantized values")
    return payload, result


@torch.no_grad()
def exact_c0_payload(wrapper):
    """Encode the existing cache, requiring bitwise reconstruction, never refit."""
    if wrapper.cached_weight is None:
        wrapper.freeze_weight("cpu")
    cache = wrapper.cached_weight.detach().float().cpu()
    scale = wrapper.wq.scale.detach().float().cpu()
    if scale.shape != (cache.shape[0], 1):
        raise ValueError("C0 has a different official weight quantizer")
    codes = (cache / scale).round().clamp(-8, 7).to(torch.int8)
    payload = _make_payload(codes, scale, torch.ones(cache.shape[1]), is_c0=True)
    if not torch.equal(decode_payload(payload), cache):
        raise AssertionError("C0 integer payload fails exact frozen-cache reconstruction")
    return payload


def split_rows(rows, config):
    if len(rows) != config["pool_size"]:
        raise ValueError("Insufficient calibration reservoir; no smaller-pool fallback")
    indices = torch.randperm(len(rows), generator=torch.Generator().manual_seed(config["seed"]))
    cut = config["fit_rows"]
    fit_ids, select_ids = indices[:cut], indices[cut:]
    if len(set(fit_ids.tolist()) & set(select_ids.tolist())):
        raise AssertionError("Fit/select overlap")
    return (
        rows.index_select(0, fit_ids.to(rows.device)),
        rows.index_select(0, select_ids.to(rows.device)),
        {
            "fit_row_index_sha256": tensor_hash(fit_ids),
            "select_row_index_sha256": tensor_hash(select_ids),
            "fit_rows": len(fit_ids),
            "select_rows": len(select_ids),
            "row_indices_disjoint": True,
            "request_disjoint": False,
        },
    )


def channel_scales(fit, config):
    observed = fit.square().mean(0) > 0
    magnitudes = {"rms": fit.square().mean(0).sqrt(), "mean_abs": fit.abs().mean(0)}
    for family in config["families"]:
        for alpha in config["alphas"]:
            if family == "mean_abs" and alpha == 0:
                continue
            scale = magnitudes[family].clamp_min(1e-8).pow(alpha).clamp_min(1e-8)
            selected = scale[observed]
            if selected.numel():
                scale = scale / (selected.amin() * selected.amax()).sqrt().clamp_min(1e-12)
            scale = torch.where(observed, scale.clamp(1e-4, 1e4), torch.ones_like(scale))
            yield family, alpha, scale


@torch.no_grad()
def ridge_delta(weight, fit, z, scale, *, ridge=0.1, output_chunk=512, force_domain=None):
    """Fit A4 error only; W4 is applied afterward and tested on select rows.

    B=W*scale, E=fit@W.T - z@B.T, lambda=ridge*||z||²/n.
    D=E.T z (z.T z+lambda I)^-1, or E.T(zz.T+lambda I)^-1z.
    The same lambda is used whichever smaller exact system is solved.
    """
    n, d = z.shape
    domain = force_domain or ("primal" if d <= n else "dual")
    if domain not in ("primal", "dual"):
        raise ValueError("Unknown ridge solve domain")
    b = weight * scale
    base = z.square().sum() / n
    if not bool(base > 0):
        return torch.zeros_like(b), {
            "solve_domain": domain,
            "lambda": 0.0,
            "ridge_multiplier": 1.0,
            "normal_equation_relative_residual": 0.0,
        }
    regularization = ridge * base
    gram = z.T @ z if domain == "primal" else z @ z.T
    eye = torch.eye(len(gram), device=gram.device, dtype=gram.dtype)
    factor = None
    for multiplier in (1.0, 10.0, 100.0):
        system = gram + regularization * multiplier * eye
        candidate, info = torch.linalg.cholesky_ex(system)
        if int(info.item()) == 0:
            factor = candidate
            break
    if factor is None:
        raise ValueError("Ridge factorization failed at all frozen numerical recovery multipliers")
    pieces, residual_max = [], 0.0
    for start in range(0, len(weight), output_chunk):
        w_part = weight[start : start + output_chunk]
        error = F.linear(fit, w_part) - F.linear(z, w_part * scale)
        rhs = z.T @ error if domain == "primal" else error
        solution = torch.cholesky_solve(rhs, factor)
        relative = torch.linalg.vector_norm(system @ solution - rhs) / torch.linalg.vector_norm(
            rhs
        ).clamp_min(1e-30)
        residual_max = max(residual_max, float(relative))
        pieces.append(solution.T if domain == "primal" else solution.T @ z)
    delta = torch.cat(pieces)
    if not torch.isfinite(delta).all():
        raise ValueError("Nonfinite ridge correction")
    return delta, {
        "solve_domain": domain,
        "system_dimension": len(gram),
        "lambda": float(regularization * multiplier),
        "ridge_multiplier": multiplier,
        "lambda_normalizer": "fit_rows regardless of primal/dual",
        "normal_equation_relative_residual": residual_max,
    }


def apply_cap(base, delta, cap):
    norm = torch.linalg.vector_norm(base)
    delta_norm = torch.linalg.vector_norm(delta)
    raw_relative = delta_norm / norm.clamp_min(1e-30)
    multiplier = torch.ones_like(norm)
    if cap is not None:
        multiplier = torch.minimum(multiplier, cap * norm / delta_norm.clamp_min(1e-30))
    result = base + delta * multiplier
    applied = torch.linalg.vector_norm(result - base) / norm.clamp_min(1e-30)
    if cap is not None and float(applied) > cap + 2e-6:
        raise AssertionError("Pre-quantization Frobenius trust cap violated")
    return result, {
        "cap": cap,
        "cap_domain": "FP32 scaled rotated target before W4",
        "uncapped_relative_correction": float(raw_relative),
        "applied_relative_correction": float(applied),
        "cap_multiplier": float(multiplier),
        "post_weight_quantization_bound_claimed": False,
    }


@torch.no_grad()
def output_score(x, z, weight, quantized, *, row_chunk=1024, output_chunk=512):
    numerator = denominator = 0.0
    for row in range(0, len(x), row_chunk):
        xx, zz = x[row : row + row_chunk], z[row : row + row_chunk]
        for out in range(0, len(weight), output_chunk):
            reference = F.linear(xx, weight[out : out + output_chunk])
            error = F.linear(zz, quantized[out : out + output_chunk]) - reference
            numerator += float(error.double().square().sum())
            denominator += float(reference.double().square().sum())
    score = numerator / max(denominator, 1e-30)
    if not math.isfinite(score):
        raise ValueError("Nonfinite select reconstruction score")
    return score


def _payload_identity(payload):
    return _json_hash(
        {k: tensor_hash(payload[k]) for k in ("packed_codes", "row_scale", "input_scale")}
        | {"shape": payload["shape"], "is_exact_c0": payload["is_exact_c0"]}
    )


@torch.no_grad()
def search_layer(wrapper, native_rows, c0, config):
    device = wrapper.weight.device
    with torch.autocast(device.type, enabled=False):
        rows = rotate_inputs(wrapper, native_rows.to(device=device, dtype=torch.float32))
        fit, select, _ = split_rows(rows, config)
        del rows
        weight = wrapper.rotated_weight().detach()
        official, plan = wrapper.owner_ref().official, wrapper.plan
        q0 = decode_payload(c0, device=device)
        z0 = activation_qdq(official, plan, select, row_chunk=config["row_chunk"])
        score0 = output_score(
            select,
            z0,
            weight,
            q0,
            row_chunk=config["row_chunk"],
            output_chunk=config["output_chunk"],
        )
        best = {"score": score0, "payload": c0, "candidate": "C0"}
        cap = config["correction_cap"]
        for family, alpha, scale in channel_scales(fit, config):
            base = weight * scale
            z_select = activation_qdq(official, plan, select / scale, row_chunk=config["row_chunk"])
            payload, q_base = quantize_payload(
                official, base, scale, row_chunk=config["weight_quantizer_row_chunk"]
            )
            variants = [("no_comp", payload, q_base)]
            if cap is not None:
                z_fit = activation_qdq(official, plan, fit / scale, row_chunk=config["row_chunk"])
                delta, _ = ridge_delta(
                    weight,
                    fit,
                    z_fit,
                    scale,
                    ridge=config["ridge"],
                    output_chunk=config["output_chunk"],
                )
                del z_fit
                target, _ = apply_cap(base, delta, cap)
                del delta
                q_payload, q_weight = quantize_payload(
                    official, target, scale, row_chunk=config["weight_quantizer_row_chunk"]
                )
                variants.append(("bounded", q_payload, q_weight))
                del target, q_payload, q_weight
            for kind, candidate_payload, quantized in variants:
                score = output_score(
                    select,
                    z_select,
                    weight,
                    quantized,
                    row_chunk=config["row_chunk"],
                    output_chunk=config["output_chunk"],
                )
                # Preserve the candidate order and prefer the reference on exact ties.
                if score < best["score"]:
                    best = {
                        "score": score,
                        "payload": candidate_payload,
                        "candidate": f"{family}:{alpha:g}:{kind}",
                    }
            del variants
        return best


def collect_margin_pools(adapter, session, requests, out):
    """Collect the 16K native decision-margin reservoir for each Linear."""
    from .decision_margin import DecisionMarginCollector

    if len(requests) != 128:
        raise ValueError("Expected 128 native calibration requests")
    pool_size = DEFAULT_CONFIG["pool_size"]
    seed = DEFAULT_CONFIG["seed"]
    collector = DecisionMarginCollector(
        adapter,
        modules=session.originals,
        reservoir_size=pool_size,
        seed=seed,
    )
    with session.mode(enabled=False, quantized=False), session.no_cache(), collector:
        for index, request in enumerate(requests, start=1):
            collector.collect(request)
            _json(
                out / "calibration-progress.json",
                {"requests": index, "total_requests": len(requests)},
            )
        # Keep references to the saved rows without cloning the model-wide pools.
        pools = {name: item.reservoir for name, item in collector._accumulators.items()}
        metadata = collector.metadata()
    if set(pools) != set(session.originals):
        raise ValueError("Some selected Linear layers were never observed")
    for name, rows in pools.items():
        if rows is None or len(rows) != pool_size or not torch.isfinite(rows).all():
            raise ValueError(f"Expected a full finite {pool_size}-row pool at {name}")
    _json(out / "calibration-summary.json", metadata)
    return pools


@torch.no_grad()
def fit_payloads(session, pools, out, *, correction_cap, config=None):
    """Fit the fixed benchmark arm and persist packed weights one layer at a time."""
    if correction_cap not in (None, 0.05, 0.10):
        raise ValueError("Expected a published compensation cap or disabled compensation")
    config = _configuration(config)
    config["correction_cap"] = correction_cap
    out = Path(out)
    target = out / "payloads"
    target.mkdir()
    started = time.perf_counter()
    layers = {}
    for name, wrapper in session.wrappers.items():
        c0 = exact_c0_payload(wrapper)
        selected = search_layer(wrapper, pools.pop(name), c0, config)
        payload = selected["payload"]
        digest = _payload_identity(payload)
        target_path = target / f"{digest}.pt"
        if not target_path.exists():
            torch.save(payload, target_path)
        layers[name] = {
            "payload_sha256": digest,
            "file_sha256": _file_hash(target_path),
            "weight_sha256": tensor_hash(wrapper.weight),
            "bias_sha256": None if wrapper.bias is None else tensor_hash(wrapper.bias),
            "plan": wrapper.plan,
            "candidate": selected["candidate"],
            "select_mse": selected["score"],
        }
        wrapper.cached_weight = None
        wrapper.scale_version = None
        _json(
            out / "fit-progress.json",
            {
                "completed_layers": len(layers),
                "total_layers": len(session.wrappers),
                "last_layer": name,
                "seconds": time.perf_counter() - started,
            },
        )
    manifest = {
        "format": "macq.packed-w4a4.v1",
        "config": config,
        "layers": layers,
        "rotation_sha256": _rotation_hash(session),
        "backend": session.official.receipt,
        "execution": "floating-point QDQ",
    }
    _json(out / "payload-manifest.json", manifest)
    return manifest


def activate_payloads(session, out):
    """Bind packed weights to the same native checkpoint and rotation state."""
    out = Path(out)
    manifest = json.loads((out / "payload-manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != "macq.packed-w4a4.v1":
        raise ValueError("Unsupported MACQ payload format")
    if manifest["rotation_sha256"] != _rotation_hash(session):
        raise ValueError("MACQ payload rotation mismatch")
    if manifest["backend"] != session.official.receipt:
        raise ValueError("MACQ payload backend mismatch")
    if set(manifest["layers"]) != set(session.wrappers):
        raise ValueError("MACQ payload layer scope mismatch")
    for name, wrapper in session.wrappers.items():
        entry = manifest["layers"][name]
        bias = None if wrapper.bias is None else tensor_hash(wrapper.bias)
        if (
            entry["plan"] != wrapper.plan
            or entry["weight_sha256"] != tensor_hash(wrapper.weight)
            or entry["bias_sha256"] != bias
        ):
            raise ValueError(f"MACQ payload does not match native layer: {name}")
        digest = entry["payload_sha256"]
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"Invalid payload digest: {name}")
        payload_path = out / "payloads" / f"{digest}.pt"
        if _file_hash(payload_path) != entry["file_sha256"]:
            raise ValueError(f"MACQ payload file hash mismatch: {name}")
        payload = torch.load(payload_path, map_location="cpu", weights_only=True)
        if _payload_identity(payload) != digest:
            raise ValueError(f"MACQ payload tensor hash mismatch: {name}")
        if payload["is_exact_c0"]:
            wrapper.cached_weight = decode_payload(payload)
        else:
            wrapper.packed_payload = payload
    session.quantized = True
    return manifest

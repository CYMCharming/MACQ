"""Orthogonal coordinate transforms for native decision model linears."""

from __future__ import annotations
import ast
import contextlib
import hashlib
import inspect
import json
import math
import random
import time
import weakref
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


class RotationBackend:
    """Load the pinned optimizer, quantizers, and Hadamard implementation."""

    def __init__(self, root):
        root = Path(root)
        pins = json.loads(
            Path(__file__).with_name("rotation-source.json").read_text(encoding="utf-8")
        )
        self.receipt = {"commit": pins["commit"], "repository": pins["repository"], "sources": {}}

        def checked(relative):
            path = root / relative
            digest = sha(path)
            if digest != pins["source_sha256"][relative]:
                raise ValueError(f"Rotation backend hash mismatch: {relative}")
            self.receipt["sources"][relative] = digest
            return path

        checked("LICENSE")
        path = checked("train_utils/optimizer.py")
        env = {"__name__": "_macq_rotation_optimizer", "__file__": str(path)}
        exec(compile(path.read_bytes(), str(path), "exec"), env)
        self.SGDG = env["SGDG"]

        path = checked("utils/quant_utils.py")
        names = {
            "get_minq_maxq",
            "asym_quant",
            "asym_dequant",
            "asym_quant_dequant",
            "sym_quant",
            "sym_dequant",
            "sym_quant_dequant",
            "STEQuantize",
            "AsymSTEQuantize",
            "ActQuantizer",
            "WeightQuantizer",
        }
        tree = ast.parse(path.read_bytes())
        nodes = [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
        ]
        if {node.name for node in nodes} != names:
            raise ValueError("Rotation backend quantizer definitions differ")
        env = {"torch": torch, "math": math}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), env)
        self.WeightQuantizer = env["WeightQuantizer"]
        self.ActQuantizer = env["ActQuantizer"]

        path = checked("utils/hadamard_utils.py")
        nodes = [
            node for node in ast.parse(path.read_bytes()).body if isinstance(node, ast.FunctionDef)
        ]
        env = {"torch": torch, "math": math}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), env)
        self.hadamard = env["matmul_hadU"]
        self.random_hadamard = env["random_hadamard_matrix"]
        self.get_hadK = env["get_hadK"]


def grouped_right(x, rotation):
    size = rotation.shape[0]
    if x.shape[-1] % size:
        raise ValueError("Head rotation does not divide native input")
    return (x.reshape(*x.shape[:-1], -1, size) @ rotation).reshape(x.shape)


def online_hadamard(x, plan, official):
    partition = plan.get("had_partition")
    if partition:
        if sum(partition) != x.shape[-1]:
            raise ValueError("Hadamard partition must cover all coordinates exactly")
        return torch.cat(
            [official.hadamard(part) for part in torch.split(x, partition, dim=-1)], dim=-1
        )
    block = plan.get("had_block_size")
    if block is None:
        return official.hadamard(x)
    return official.hadamard(x.reshape(*x.shape[:-1], -1, block)).reshape(x.shape)


class RotatedLinear(nn.Module):
    def __init__(self, original, owner, name, plan):
        super().__init__()
        self.original = original
        # A weak reference guarantees exactly one registered shared R1 parameter.
        object.__setattr__(self, "owner_ref", weakref.ref(owner))
        self.name = name
        self.plan = plan
        self.wq = owner.official.WeightQuantizer()
        self.wq.configure(4, perchannel=True, sym=True, mse=True, weight_groupsize=-1)
        self.aq = owner.official.ActQuantizer()
        # Official asym per-token activation recipe; output projection groups=head.
        self.aq.configure(4, groupsize=plan.get("head_dim", -1), sym=False, clip_ratio=1.0)
        self.cached_weight = None
        self.scale_version = None
        self.packed_payload = None

    @property
    def weight(self):
        return self.original.weight

    @property
    def bias(self):
        return self.original.bias

    @property
    def in_features(self):
        return self.original.in_features

    @property
    def out_features(self):
        return self.original.out_features

    def rotated_weight(self):
        owner = self.owner_ref()
        w = self.original.weight.float()
        if self.plan["input_r1"]:
            w = w @ owner.R1
        if self.plan.get("r2_key"):
            w = grouped_right(w, owner.R2[self.plan["r2_key"]])
        if self.plan["online_had"]:
            w = online_hadamard(w, self.plan, owner.official)
        if self.plan["output_r1"]:
            w = owner.R1.T @ w
        return w

    def freeze_weight(self, device="cpu"):
        with torch.no_grad(), torch.autocast(self.weight.device.type, enabled=False):
            w = self.rotated_weight()
            self.wq.find_params(w)
            self.cached_weight = self.wq.quantize(w).to(device).clone()
        return self.cached_weight.numel() * self.cached_weight.element_size()

    def forward(self, x):
        owner = self.owner_ref()
        if not owner.enabled:
            return self.original(x)
        # Match nn.Linear's native OUTPUT dtype, not merely its input dtype:
        # an FP32 residual can enter Linear under external BF16 autocast.
        dtype = (
            torch.get_autocast_dtype(x.device.type)
            if torch.is_autocast_enabled(x.device.type)
            else x.dtype
        )
        # Explicit FP32 coordinate changes avoid BF16 rounding at every inverse.
        # The resulting QDQ matmul is FP32; output returns to native dtype.
        with torch.autocast(x.device.type, enabled=False):
            xx = x.float()
            bias = self.original.bias
            if self.plan["input_r1"]:
                xx = xx @ owner.R1
            if self.plan.get("r2_key"):
                r2 = owner.R2[self.plan["r2_key"]]
                xx = grouped_right(xx, r2)
            if self.plan["online_had"]:
                xx = online_hadamard(xx, self.plan, owner.official)
            payload = self.packed_payload if owner.quantized else None
            cached = owner.quantized and self.cached_weight is not None
            if payload is not None:
                if torch.is_grad_enabled():
                    raise RuntimeError("Packed W4A4 weights require inference or no_grad")
                from .margin_reconstruction import decode_payload

                xx = xx / payload["input_scale"].to(x.device)
                w = decode_payload(payload, device=x.device)
            elif cached:
                if torch.is_grad_enabled():
                    raise RuntimeError("Cannot train with frozen quantized weight cache")
                w = self.cached_weight.to(x.device)
            else:
                w = self.rotated_weight()
            if owner.quantized:
                # Official parameter search is not differentiated; STE is.
                with torch.no_grad():
                    if (
                        payload is None
                        and not cached
                        and self.scale_version != owner.weight_version
                    ):
                        self.wq.find_params(w.detach())
                        self.scale_version = owner.weight_version
                    self.aq.find_params(xx.detach())
                if payload is None and not cached:
                    w = self.wq.quantize(w)
                xx = self.aq(xx)
                self.aq.free()
            y = F.linear(xx, w, None)
            if self.plan["output_r1"]:
                y = y @ owner.R1.T
            if bias is not None:
                y = y + bias.float()
            return y.to(dtype)


def native_plan(backbone, modules, online_had):
    config = backbone.config
    hidden = int(getattr(config, "hidden_size"))
    plans = {}
    heads = {}
    for name, module in modules.items():
        mixer_out = name.endswith(("self_attn.o_proj", "linear_attn.out_proj", "attn.Wo"))
        mlp_out = name.endswith(("mlp.down_proj", "mlp.Wo"))
        p = {
            "input_r1": module.in_features == hidden and not mixer_out,
            "output_r1": module.out_features == hidden,
            "online_had": bool(online_had and mlp_out),
            "r2_key": None,
        }
        if mixer_out:
            parent = backbone.get_submodule(name.rsplit(".", 1)[0])
            if name.endswith("linear_attn.out_proj"):
                head_dim = getattr(
                    parent, "head_v_dim", getattr(config, "linear_value_head_dim", None)
                )
            else:
                head_dim = getattr(parent, "head_dim", getattr(config, "head_dim", None))
                if head_dim is None:
                    head_dim = hidden // int(config.num_attention_heads)
            if not head_dim or module.in_features % head_dim:
                raise ValueError("Native value-head shape unsupported: " + name)
            key = name.replace(".", "_")
            heads[key] = int(head_dim)
            p.update(
                r2_key=key,
                head_dim=int(head_dim),
                r2_location="after native gate, immediately before native output projection",
            )
        if not (p["input_r1"] or p["output_r1"] or p["r2_key"]):
            raise ValueError("Unaudited Linear projection in native backbone: " + name)
        plans[name] = p
    if not heads:
        raise ValueError("No native attention/value-head output projection identified")
    return hidden, plans, heads


class RotationSession(nn.Module):
    def __init__(
        self,
        backbone,
        official_root,
        *,
        online_had=False,
        seed=0,
        modules=None,
        protected_model=None,
        checkpoint_blocks=True,
    ):
        super().__init__()
        self.official = RotationBackend(official_root)
        self.online_had = online_had
        self.enabled = True
        self.quantized = False
        self.closed = False
        self.weight_version = 0
        self.backbone_ref = weakref.ref(backbone)
        if modules is None:
            from s1q.quantization import selected_linear_modules

            modules = selected_linear_modules(backbone)
        if not modules:
            raise ValueError("No quantizable native Linear layers were found")
        self.originals = dict(modules)
        hidden, self.plans, heads = native_plan(backbone, modules, online_had)
        for name, plan in self.plans.items():
            if not plan["online_had"]:
                continue
            width = modules[name].in_features
            try:
                self.official.get_hadK(width)
                plan.update(had_block_size=None, had_mode="official full-width R4")
            except AssertionError:
                remaining = width
                partition = []
                while remaining:
                    block = 1 << (remaining.bit_length() - 1)
                    partition.append(block)
                    remaining -= block
                plan.update(
                    had_partition=partition,
                    had_mode="native width adaptation: greedy power-of-two direct sum of official Hadamards",
                    no_padding_or_cropping=True,
                )
        device = next(backbone.parameters()).device
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.R1 = nn.Parameter(self.official.random_hadamard(hidden, "cpu").float().to(device))
            self.R2 = nn.ParameterDict(
                {
                    k: nn.Parameter(self.official.random_hadamard(d, "cpu").float().to(device))
                    for k, d in heads.items()
                }
            )
        self.original_requires_grad = [
            (p, p.requires_grad) for p in (protected_model or backbone).parameters()
        ]
        self.wrappers = {}
        self.trace = []
        self.checkpoint_overrides = []
        try:
            for p, _ in self.original_requires_grad:
                p.requires_grad_(False)
            for name, original in self.originals.items():
                parent_name, leaf = name.rsplit(".", 1) if "." in name else ("", name)
                parent = backbone.get_submodule(parent_name) if parent_name else backbone
                wrapper = RotatedLinear(original, self, name, self.plans[name])
                self.wrappers[name] = wrapper
                setattr(parent, leaf, wrapper)
            if checkpoint_blocks:
                self.install_block_checkpoints()
        except BaseException:
            self.close()
            raise

    def install_block_checkpoints(self):
        from torch.utils.checkpoint import checkpoint

        backbone = self.backbone_ref()
        blocks = getattr(backbone, "layers", None)
        if blocks is None:
            raise ValueError("Native block list unavailable for memory-bounded autograd")
        for block in blocks:
            had_override = "forward" in block.__dict__
            old_override = block.__dict__.get("forward")
            bound = block.forward
            owner_ref = weakref.ref(self)

            def forward(*args, _bound=bound, _owner=owner_ref, **kwargs):
                owner = _owner()
                if torch.is_grad_enabled() and owner.enabled and owner.quantized:
                    if (
                        kwargs.get("past_key_values") is not None
                        or kwargs.get("cache_params") is not None
                    ):
                        raise ValueError(
                            "Stateful native cache cannot be checkpointed; use uncached full decision requests"
                        )
                    # Some native decoders temporarily select a custom causal
                    # attention implementation only inside infer(). Backward
                    # happens after that context exits, so replay its config.
                    config = owner.backbone_ref().config
                    captured = getattr(config, "_attn_implementation", None)

                    @contextlib.contextmanager
                    def replay_attention_context():
                        previous = getattr(config, "_attn_implementation", None)
                        if captured is not None:
                            config._attn_implementation = captured
                        try:
                            yield
                        finally:
                            if captured is not None:
                                config._attn_implementation = previous

                    # Non-reentrant supports frozen embedding inputs and external R1/R2.
                    return checkpoint(
                        _bound,
                        *args,
                        use_reentrant=False,
                        preserve_rng_state=True,
                        context_fn=lambda: (contextlib.nullcontext(), replay_attention_context()),
                        **kwargs,
                    )
                return _bound(*args, **kwargs)

            self.checkpoint_overrides.append((block, had_override, old_override))
            block.forward = forward

    def rotation_state(self):
        return {
            "R1": self.R1.detach().cpu().clone(),
            **{"R2." + k: v.detach().cpu().clone() for k, v in self.R2.items()},
        }

    def load_rotations(self, state):
        parameters = {"R1": self.R1, **{"R2." + key: value for key, value in self.R2.items()}}
        if not isinstance(state, dict) or state.keys() != parameters.keys():
            raise ValueError("Rotation scope differs from the native model")
        for name, tensor in state.items():
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.shape != parameters[name].shape
                or tensor.dtype != torch.float32
                or not torch.isfinite(tensor).all()
            ):
                raise ValueError(f"Invalid rotation tensor: {name}")
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(state[name])
        self.clear_weight_cache()

    def clear_weight_cache(self):
        self.weight_version += 1
        for wrapper in self.wrappers.values():
            wrapper.cached_weight = None
            wrapper.packed_payload = None
            wrapper.scale_version = None

    def freeze_weights(self, device="cpu"):
        """Exact QDQ cache: same FP32 values as dynamic official quantization.

        CPU storage bounds GPU residency to one transformed projection at a time.
        It changes neither rounding nor scales, and makes evaluation practical.
        """
        start = time.time()
        self.quantized = True
        count = sum(w.freeze_weight(device=device) for w in self.wrappers.values())
        self.cache_receipt = {
            "device": str(device),
            "bytes": count,
            "seconds": time.time() - start,
            "numeric_values": "exact FP32 QDQ weight values",
            "speedup_claim": False,
        }
        return self.cache_receipt

    def orthogonality(self):
        with torch.no_grad():
            return {
                name: float(
                    (p.float().T @ p.float() - torch.eye(len(p), device=p.device)).abs().max()
                )
                for name, p in [("R1", self.R1), *((k, v) for k, v in self.R2.items())]
            }

    @contextlib.contextmanager
    def mode(self, *, enabled=None, quantized=None):
        old = (self.enabled, self.quantized)
        if enabled is not None:
            self.enabled = enabled
        if quantized is not None:
            self.quantized = quantized
        try:
            yield
        finally:
            self.enabled, self.quantized = old

    def close(self):
        if self.closed:
            return
        backbone = self.backbone_ref()
        for block, had_override, old_override in self.checkpoint_overrides:
            if had_override:
                block.forward = old_override
            else:
                delattr(block, "forward")
        for name, original in self.originals.items():
            parent_name, leaf = name.rsplit(".", 1) if "." in name else ("", name)
            parent = backbone.get_submodule(parent_name) if parent_name else backbone
            setattr(parent, leaf, original)
        for p, requires in self.original_requires_grad:
            p.requires_grad_(requires)
        self.closed = True
        self.clear_weight_cache()

    @contextlib.contextmanager
    def no_cache(self):
        backbone = self.backbone_ref()
        config = backbone.config
        present = hasattr(config, "use_cache")
        before = getattr(config, "use_cache", None)
        had_override = "forward" in backbone.__dict__
        old_override = backbone.__dict__.get("forward")
        bound = backbone.forward
        parameters = inspect.signature(bound).parameters
        wrap_forward = present and "use_cache" in parameters
        if present:
            config.use_cache = False
        if wrap_forward:

            def forward(*args, **kwargs):
                if kwargs.get("past_key_values") is not None:
                    raise ValueError(
                        "Full-request rotation fit cannot consume a preexisting mutable cache"
                    )
                kwargs["use_cache"] = False
                if "past_key_values" in parameters:
                    kwargs["past_key_values"] = None
                return bound(*args, **kwargs)

            backbone.forward = forward
        try:
            yield
        finally:
            if present:
                config.use_cache = before
            if wrap_forward:
                if had_override:
                    backbone.forward = old_override
                else:
                    delattr(backbone, "forward")

    def fit(
        self, requests, infer, *, steps=100, lr=1.5, seed=42, out=None, gradient_accumulation=8
    ):
        # Native full-row decisions do not need their returned KV cache. Recompute
        # must never update a shared mutable cache a second time during backward.
        with self.no_cache():
            return self._fit(
                requests,
                infer,
                steps=steps,
                lr=lr,
                seed=seed,
                out=out,
                gradient_accumulation=gradient_accumulation,
            )

    def _fit(self, requests, infer, *, steps, lr, seed, out, gradient_accumulation):
        if not requests or steps <= 0:
            raise ValueError("Nonempty fit requests and positive steps required")
        if not isinstance(gradient_accumulation, int) or gradient_accumulation < 1:
            raise ValueError("Positive integer gradient accumulation required")
        self.fit_request_count = len(requests)
        random.seed(seed)
        torch.manual_seed(seed)
        self.clear_weight_cache()
        teacher = []
        with self.mode(enabled=False), torch.no_grad():
            for request in requests:
                teacher.append([v.detach().float().cpu() for v in infer(request)])
        self.quantized = True
        optimizer = self.official.SGDG([self.R1, *self.R2.values()], lr=lr, stiefel=True)
        order = []
        epoch = -1
        self.gradient_accumulation = gradient_accumulation
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            self.weight_version += 1
            indices = []
            losses = []
            for micro in range(gradient_accumulation):
                position = step * gradient_accumulation + micro
                new_epoch = position // len(requests)
                if new_epoch != epoch:
                    epoch = new_epoch
                    order = torch.randperm(
                        len(requests), generator=torch.Generator().manual_seed(seed + epoch)
                    ).tolist()
                index = order[position % len(order)]
                indices.append(index)
                logits = infer(requests[index])
                targets = [t.to(logits[0].device) for t in teacher[index]]
                loss = native_kl(logits, targets)
                if not loss.requires_grad or not torch.isfinite(loss):
                    raise ValueError("Native decision KD has no finite autograd graph")
                (loss / gradient_accumulation).backward()
                losses.append(float(loss.detach()))
                del logits, targets, loss
            grads = {
                name: None if p.grad is None else float(p.grad.float().norm())
                for name, p in [("R1", self.R1), *((k, v) for k, v in self.R2.items())]
            }
            if not grads["R1"] or any(
                v is not None and not math.isfinite(v) for v in grads.values()
            ):
                raise ValueError("Invalid rotation gradient")
            grad_norm = float(torch.nn.utils.clip_grad_norm_([self.R1, *self.R2.values()], 1.0))
            # HF Trainer cosine schedule without warmup; batch1 accumulated eight times.
            rate = lr * 0.5 * (1 + math.cos(math.pi * step / steps))
            optimizer.param_groups[0]["lr"] = rate
            optimizer.step()
            self.weight_version += 1
            row = {
                "step": step + 1,
                "request_indices": indices,
                "native_teacher_kl": sum(losses) / len(losses),
                "lr": rate,
                "effective_batch_size": gradient_accumulation,
                "gradient_norm_before_clip": grad_norm,
                "R1_gradient_norm": grads["R1"],
                "R2_nonzero_gradients": sum(
                    v is not None and v > 0 for k, v in grads.items() if k != "R1"
                ),
                "orthogonality_max_abs": max(self.orthogonality().values()),
            }
            if row["orthogonality_max_abs"] > 1e-3:
                raise ValueError("SGDG orthogonality drift >1e-3")
            self.trace.append(row)
            if out is not None:
                with (Path(out) / "rotation-trace.jsonl").open("a") as f:
                    f.write(json.dumps(row) + "\n")
                dump(
                    Path(out) / "progress.json",
                    {
                        "stage": "rotation_training",
                        "step": step + 1,
                        "steps": steps,
                        "unix": time.time(),
                    },
                )
        return teacher


def native_kl(student, teacher):
    if len(student) != len(teacher) or not student:
        raise ValueError("Native question mismatch")
    values = []
    for s, t in zip(student, teacher):
        if s.shape != t.shape:
            raise ValueError("Native candidate mismatch")
        values.append(
            F.kl_div(F.log_softmax(s.float(), -1), F.softmax(t.float(), -1), reduction="sum")
        )
    return torch.stack(values).mean()

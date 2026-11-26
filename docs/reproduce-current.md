# Reproduce MACQ

## Install and prepare inputs

Use Python 3.12 and a CUDA PyTorch build compatible with the pinned native adapters. The [model manifest](../configs/models.json) records upstream checkpoints. [CI constraints](../.github/constraints.txt) pin the CPU test environment.

```bash
python -m pip install -e '.[models,test]'
python scripts/setup_sources.py --families kev laya rotation
python scripts/prepare_data.py shared --source-root work/upstream/kev --output work/data/shared
```

`--checkpoint-dir` accepts a local checkpoint. Pass `--checkpoint-manifest` to verify its revision and weight hashes. A local Intern-Decision or StartLux-Decision checkpoint also needs the publisher's inference code and its hashes in the manifest. Load each StartLux checkpoint in its own process. `STARTLUX_ALLOW_SLOW=1` enables the reference runtime used for StartLux in this benchmark.

## Run the W4A4 benchmark

Acquire [Intern-Decision evaluation data](https://github.com/InternLM/Intern-Decision/tree/3572c8a68b5df5dafe02d0e093989ba8ec0183bc/benchmarks/accuracy-v1) at the recorded revision, then recreate the 22 suites locally.

```bash
python scripts/prepare_expanded_benchmarks.py \
  --source-root work/upstream/kev \
  --intern-source-root work/upstream/Intern-Decision/benchmarks/accuracy-v1 \
  --exclusion-registry configs/benchmarks/expanded-20261007/exclusion_registry.json \
  --output work/data/expanded --seed 20261007 --count 128 \
  --benchmark-id expanded-native-decision-20261007-v2

macq optimize --model kev-4b --data-dir work/data/shared \
  --benchmark-manifest work/data/expanded/manifest.json --output-dir work/kev4-expanded
```

The entry point uses W4A4 and fits orthogonal rotations for 100 updates with an effective batch size of eight native calibration requests. For exact replay with an existing rotation artifact, pass `--rotation-checkpoint /path/to/trained-rotations.pt` and compare its hash with the benchmark record. Check the per-suite file hashes against the frozen manifest. The local manifest can differ in provenance fields. [Frozen benchmark suite manifest](../configs/benchmarks/expanded-20261007/manifest.json) · [Calibration, model, and rotation hashes](../benchmarks/protocol.json)

The runner writes packed weights, learned rotations, and a payload manifest in a fresh output directory. With `--benchmark-manifest` it also writes `scores.json` and per-question predictions after the fit. Exact scores depend on the recorded checkpoint, runtime, and dataset admission.

## Python API

```python
import torch

from s1q.api import load_macq, quantize_macq
from s1q.data import load_records
from s1q.models import load_model

adapter = load_model('kev-4b', source_dir='work/upstream')
adapter.model.eval()
calibration = load_records('work/data/shared/calibration.jsonl')
with quantize_macq(
    adapter, calibration, rotation_source='work/upstream/rotation',
    output_dir='work/kev4-packed',
), torch.inference_mode():
    logits = adapter.infer(calibration[0])

with load_macq(
    adapter, 'work/kev4-packed', rotation_source='work/upstream/rotation',
), torch.inference_mode():
    reloaded_logits = adapter.infer(calibration[0])
```

The fit accepts 128 admitted requests and strips labels before computing margins. The context managers restore the original model on exit. Dependency revisions and source hashes are checked at runtime. [Third-party notices](../NOTICE)

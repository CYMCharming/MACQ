# Reproduce MACQ

## Install and prepare inputs

Use Python 3.12 and a CUDA PyTorch build compatible with the pinned native adapters. The [model manifest](../configs/models.json) records upstream checkpoints. [CI constraints](../.github/constraints.txt) pin the CPU test environment.

```bash
python -m pip install -e '.[models,test]'
python scripts/setup_sources.py --families kev laya
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

python scripts/run_expanded_benchmarks.py \
  --model kev-4b --data-dir work/data/shared --benchmark-manifest work/data/expanded/manifest.json \
  --output-dir work/kev4-expanded --methods macq,rtn --bits 4 --activation-bits 4 \
  --group-size 128 --calibration-count 128 --reservoir-size 16384 --seed 20261004
```

Compare the per-suite file hashes against the [frozen benchmark manifest](../configs/benchmarks/expanded-20261007/manifest.json). The local manifest can differ in provenance fields. The [score manifest](../benchmarks/protocol.json) records the calibration hash and checkpoint manifests used for the published result table. The evaluator writes the configuration, model and dataset identities, per-question logits, and per-suite summaries to the run directory. Exact scores depend on the recorded checkpoint, runtime and dataset admission.

## Python API

```python
from s1q.api import quantize_macq
from s1q.data import load_records
from s1q.models import load_model

adapter = load_model('kev-4b', source_dir='work/upstream')
adapter.model.eval()
calibration = load_records('work/data/shared/calibration.jsonl')
session, details = quantize_macq(adapter, calibration)
with session:
    logits = adapter.infer(calibration[0])
```

The API accepts already admitted calibration requests. The context manager restores the original model when it closes. For a smaller budget, pass `reservoir_size` explicitly. The `s1q` executable and import paths remain available for existing scripts.

# MACQ

**Margin-aware Activation Compensation Quantization of Decision Models**

MACQ is a post-training quantization method for open Jev-like decision models. It keeps the native question compiler and decision readout, samples calibration tokens using the model's top-two decision margins, and selects low-bit weights by reconstructing outputs with quantized weights and activations. Bounded activation-error corrections are folded into the quantized layer weights.

[中文说明](README.zh-CN.md) · [Method](docs/method.md) · [Reproduce](docs/reproduce-current.md) · [W4A4 results](docs/results.md)

## Quick start

Use Python 3.12 and a CUDA PyTorch build compatible with your model runtime. Install the package and prepare native calibration and evaluation inputs.

```bash
python -m pip install -e '.[models]'
python scripts/setup_sources.py --families kev rotation
python scripts/prepare_data.py shared --source-root work/upstream/kev --output work/data/shared
macq optimize --model kev-4b --data-dir work/data/shared --output-dir work/kev4-macq
```

`macq optimize` fits and saves the W4A4 profile with 16,384 sampled token rows per quantized layer from 128 calibration requests. The reproduction guide also covers the frozen evaluation suites, precomputed rotations, and reloading packed weights. [Reproduction guide](docs/reproduce-current.md)

## Main results

Accuracy averages 22 evaluation suites within each of 11 models, then weights the models equally. Each model covers 2,871 decisions.

| Native BF16 | RTN (W4A4) | MACQ@16K (W4A4) |
| ---: | ---: | ---: |
| 77.44% | 35.12% | **74.82%** |

The [per-model comparison](docs/results.md) covers Kev, Intern-Decision, StartLux-Decision and Laya. It includes the baseline settings and links to the full aggregate data and per-suite counts. The evaluator uses floating-point quantize–dequantize execution.

## License

Apache-2.0 for this repository. See [NOTICE](NOTICE) and the [upstream licenses](licenses/README.md) for model and source attribution.

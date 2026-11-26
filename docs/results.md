# W4A4 benchmark

Scores are accuracy percentages. Each model score is the unweighted mean over 22 pinned evaluation suites (2,871 decisions per model). The final row weights all 11 models equally. MACQ used a 16K-row reservoir per layer.

| Model | Native BF16 | RTN | AWQ + A4 | GPTQ + A4 | OmniQuant (Linear) | SpinQuant (Had) | MACQ@16K |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Kev-0.8B | 70.13 | 34.63 | 39.17 | 32.69 | 53.28 | 66.91 | 59.17 |
| Kev-4B | 83.44 | 32.04 | 36.42 | 36.13 | 62.69 | 81.26 | 66.01 |
| Kev-9B | 83.23 | 35.21 | 32.79 | 35.24 | 62.11 | 81.06 | 67.84 |
| Intern-Decision-0.8B | 64.95 | 36.11 | 36.20 | 35.37 | 32.04 | 60.59 | 44.24 |
| Intern-Decision-2B | 72.44 | 34.05 | 36.63 | 34.33 | 37.17 | 66.05 | 50.68 |
| Intern-Decision-4B | 80.63 | 33.10 | 35.38 | 35.64 | 50.20 | 78.94 | 59.12 |
| StartLux-Decision-0.8B | 80.62 | 33.55 | 34.71 | 33.25 | 50.79 | 75.77 | 61.46 |
| StartLux-Decision-2B | 82.35 | 34.41 | 36.31 | 34.39 | 37.82 | 79.38 | 63.75 |
| StartLux-Decision-4B | 85.79 | 34.49 | 35.65 | 36.32 | 60.80 | 84.75 | 68.93 |
| StartLux-Decision-9B | 88.08 | 35.09 | 35.78 | 35.58 | 73.52 | 86.50 | 77.74 |
| Laya | 60.13 | 43.65 | 50.23 | 44.35 | 50.19 | 59.16 | 52.77 |
| **Mean (11 models)** | **77.44** | **35.12** | **37.21** | **35.75** | **51.87** | **74.58** | **61.06** |

[All 12 methods and references](../benchmarks/w4a4-models.csv) · [Per-suite counts and accuracies](../benchmarks/w4a4-suites.csv) · [Protocol hashes](../benchmarks/protocol.json)

## Comparison protocol

- W4A4 methods use matched model identities and evaluation suites. Calibration-based methods share 128 native requests. RTN and HQQ use calibration-free weight quantization.
- AWQ, GPTQ, HQQ, and AdaRound use their weight quantization procedures with an A4 activation path. SmoothQuant uses a W4A4 adaptation. OmniQuant and SpinQuant baselines are adapted to native model architectures. Group sizes and activation scopes follow each implementation.
- These are fixed research evaluation suites, including previously inspected data and a 48-item JevBench subset. Evaluation outputs are separate from calibration and candidate selection. Accuracy measurements use floating-point quantize–dequantize execution.

See the [reproduction guide](reproduce-current.md) for input materialization and the MACQ command.

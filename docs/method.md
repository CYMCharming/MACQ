# MACQ method

MACQ quantizes selected backbone linear layers at W4A4. It preserves native embeddings, normalization, non-linear recurrence, and the typed decision readout. Kev adapters are merged into the backbone before quantization.

## Decision-margin sampling

For each calibration question, take the two highest native logits and backpropagate their margin divided by one plus its detached magnitude. For a token at a quantized layer, sum the squared gradient norms over questions from the same request. Normalize the energies per request and layer, shrink toward uniform, and bound the resulting importance weights. The sampler uses these weights as reservoir priorities. The reconstruction objective gives the sampled rows equal weight.

The benchmark uses 128 label-free native requests and up to 16,384 rows per layer. MACQ shuffles the saved rows deterministically and splits them into fit and selection halves. Both halves come from calibration requests.

## Joint weight and activation scoring

For input rows $X$, layer weight $W$, and positive input-channel scale $s$, let $B=W\operatorname{diag}(s)$ and $Z=Q_A(X\operatorname{diag}(s)^{-1})$. Candidates are scored on selection rows using

$$
\frac{\lVert Z_{\mathrm{select}}\widehat B^{\mathsf T}-X_{\mathrm{select}}W^{\mathsf T}\rVert_F^2}
{\max(\lVert X_{\mathrm{select}}W^{\mathsf T}\rVert_F^2,\epsilon)}.
$$

$\widehat B$ is reconstructed from the actual groupwise weight quantizer. W4 uses symmetric groups of 128 weights and A4 uses dynamic per-token quantization. The search uses RMS and mean-absolute fit-row channel statistics, two clipping settings, and corrected and uncorrected weights. It evaluates 44 candidates per layer at W4A4.

## Bounded compensation

On fit rows, let $E=X_{\mathrm{fit}}W^{\mathsf T}-Z_{\mathrm{fit}}B^{\mathsf T}$. The fit solves a ridge regression to reconstruct $E$ from the quantized activation input $Z_{\mathrm{fit}}$. The correction has Frobenius norm at most $0.10\lVert B\rVert_F$. Candidates quantize the original rescaled weight and the rescaled weight with bounded correction. Accepted corrections are folded into the quantized weights, and the selection half chooses the final candidate.

The margin gradient is a token sampling signal. Prior work on gradient-guided reconstruction includes [GuidedQuant](https://proceedings.mlr.press/v267/kim25d.html). [ERQ](https://proceedings.mlr.press/v235/zhong24a.html) describes ridge-based activation error compensation, and [GPTAQ](https://arxiv.org/abs/2504.02692) also reconstructs native outputs under mixed weight and activation errors.

The [W4A4 benchmark](results.md) reports decision accuracy with floating-point QDQ on fixed evaluation inputs.

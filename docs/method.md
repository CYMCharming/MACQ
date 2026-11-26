# MACQ method

MACQ quantizes selected backbone linear layers at W4A4. It preserves native embeddings, normalization, non-linear recurrence, and the typed decision readout. Kev adapters are merged into the backbone before quantization. The benchmark execution profile uses frozen orthogonal transforms on the selected projections.

## Decision-margin sampling

For each calibration question, take the two highest native logits and backpropagate their margin divided by one plus its detached magnitude. For a token at a quantized layer, sum the squared gradient norms over questions from the same request. Normalize the energies per request and layer, shrink toward uniform, and bound the resulting importance weights. The sampler uses these weights as reservoir priorities. The reconstruction objective gives the sampled rows equal weight.

The benchmark uses 128 native requests with labels stripped. For every selected layer, MACQ transforms the 16,384 sampled native rows into its frozen coordinates, then deterministically partitions the rows into 8,192 fit and 8,192 selection rows. The two sets of rows can come from the same requests.

## Joint weight and activation scoring

For input rows $X$ and layer weight $W$ in the frozen coordinates, let $B=W\operatorname{diag}(s)$ and $Z=Q_A(X\operatorname{diag}(s)^{-1})$ for positive input-channel scale $s$. Candidates are scored on selection rows using

$$
\frac{\lVert Z_{\mathrm{select}}\widehat B^{\mathsf T}-X_{\mathrm{select}}W^{\mathsf T}\rVert_F^2}
{\max(\lVert X_{\mathrm{select}}W^{\mathsf T}\rVert_F^2,\epsilon)}.
$$

$\widehat B$ is reconstructed from symmetric per-output-channel W4 codes with MSE clipping. A4 uses dynamic asymmetric quantization with native value-head grouping at mixer outputs. The search evaluates 11 channel scales from RMS and mean-absolute fit-row statistics and includes the exact quantized reference in its candidate set. With compensation enabled it evaluates 23 candidates per layer, including the 11 uncompensated candidates. With compensation disabled it evaluates 12 candidates.

## Bounded compensation

On fit rows, let $E=X_{\mathrm{fit}}W^{\mathsf T}-Z_{\mathrm{fit}}B^{\mathsf T}$. The fit solves a ridge regression to reconstruct $E$ from $Z_{\mathrm{fit}}$ with penalty $0.1\lVert Z_{\mathrm{fit}}\rVert_F^2/n_{\mathrm{fit}}$. The default cap limits the Frobenius norm of the correction to $0.10\lVert B\rVert_F$ before requantization. StartLux-Decision-0.8B uses $0.05$ and the 2B and 4B models use the uncompensated candidate set. The selection rows choose the final quantized weight for each layer.

The margin gradient is a token sampling signal. Prior work on gradient-guided reconstruction includes [GuidedQuant](https://proceedings.mlr.press/v267/kim25d.html). [ERQ](https://proceedings.mlr.press/v235/zhong24a.html) describes ridge-based activation error compensation, and [GPTAQ](https://arxiv.org/abs/2504.02692) also reconstructs native outputs under mixed weight and activation errors.

The [W4A4 benchmark](results.md) reports decision accuracy with floating-point QDQ on fixed evaluation inputs.

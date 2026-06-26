# rsomics-boxcox

Box-Cox power transform, profile log-likelihood, and optimal-lambda search —
value-exact to `scipy.stats.boxcox` / `boxcox_llf` / `boxcox_normmax`
(scipy 1.17.1), faster single-threaded than scipy on large inputs.

## What it does

Given a strictly positive sample (one value per line), it can:

- **Transform** for a fixed lambda: `y = (xᵏ − 1)/λ` for λ≠0, `ln x` for λ=0.
- **Find the optimal lambda** (no `--lambda`) by maximizing the Box-Cox
  log-likelihood (MLE) or the probability-plot correlation (`pearsonr`), via
  Brent's method over the default `(-2, 2)` bracket — then transform with it.
- **Evaluate the log-likelihood** at a given lambda with `--llf-at`.

```text
rsomics-boxcox <DATA> [--lambda L] [--llf-at L] [--normmax-method mle|pearsonr]
```

Output is one transformed value per line, preceded by a `# lambda <L>` header.
With `--llf-at L` only the log-likelihood at `L` is printed. `--json` emits a
single result envelope.

## Examples

```sh
# Optimal lambda by MLE, then transform
rsomics-boxcox sample.tsv

# Transform for a fixed lambda
rsomics-boxcox sample.tsv --lambda 0.3

# Log-likelihood at a lambda
rsomics-boxcox sample.tsv --llf-at 0.5

# Optimal lambda by the Pearson probability-plot correlation
rsomics-boxcox sample.tsv --normmax-method pearsonr
```

## Accuracy

Against scipy 1.17.1 on the committed goldens: the transform is **bit-exact**
for a given lambda; the log-likelihood matches to ≤ 1e-12 relative error; the
MLE / Pearson optimal lambda matches to ≤ 1e-6 (an iterative optimizer's path
tolerance — see Boundaries).

## Boundaries

The optimal lambda is the value Brent's method converges to. Its bracket search
and parabolic/golden-section iteration are ported faithfully, so the lambda
matches scipy to within an optimizer's tolerance (observed ≤ ~2e-8), not to the
last bit — this is the standard value-exact bound for an iterative optimizer.
The transform at *any* fixed lambda is bit-exact.

scipy's `ymax` overflow guard (which would substitute a constrained lambda for
near-degenerate data driving lambda to extremes) is detected and reported as an
error rather than silently returning a different lambda; on realistic data it
never fires.

## Origin

This crate is an independent Rust reimplementation of `scipy.stats.boxcox`,
`boxcox_llf`, and `boxcox_normmax` based on:

- G.E.P. Box and D.R. Cox, "An Analysis of Transformations", Journal of the
  Royal Statistical Society B, 26, 211-252 (1964).
- J.J. Filliben, "The Probability Plot Correlation Coefficient Test for
  Normality", Technometrics 17, 111-117 (1975), for the uniform order-statistic
  medians used by the `pearsonr` method.
- The scipy 1.17.1 source (`scipy/stats/_morestats.py`, `scipy/optimize`,
  cephes `boxcox`/`ndtri`), which is BSD-3-licensed and may be read and cited.

The cephes `expm1(λ·ln x)/λ` transform, the log-space variance in the
log-likelihood, and the `optimize.brent` bracket + iteration are reproduced to
keep results value-exact. Test fixtures are generated independently and the
scipy reference outputs are committed, so the compat test runs without scipy.

License: MIT OR Apache-2.0.
Upstream credit: SciPy (https://scipy.org, BSD-3-Clause).

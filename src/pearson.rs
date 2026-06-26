//! Pearson-correlation objective for `boxcox_normmax(method='pearsonr')`.
//!
//! scipy maximizes the correlation between the sorted transformed data and the
//! normal quantiles of the uniform order-statistic medians (a probability-plot
//! correlation). The objective handed to the optimizer is `1 − r`.

use crate::ndtri::ndtri;
use crate::sum::pairwise_sum;
use crate::transform::boxcox;

/// Filliben uniform order-statistic medians (`_calc_uniform_order_statistic_medians`).
pub fn uniform_order_statistic_medians(n: usize) -> Vec<f64> {
    let nf = n as f64;
    let mut v = vec![0.0; n];
    let last = 0.5_f64.powf(1.0 / nf);
    v[n - 1] = last;
    v[0] = 1.0 - last;
    // scipy fills v[1..n-1] with `(i - 0.3175)/(n + 0.365)` for i in 2..n.
    for (idx, vi) in v.iter_mut().enumerate().take(n - 1).skip(1) {
        let i = (idx + 1) as f64;
        *vi = (i - 0.3175) / (nf + 0.365);
    }
    v
}

/// Normal quantiles `norm.ppf(medians)` — the x-axis of the probability plot.
pub fn normal_quantiles(n: usize) -> Vec<f64> {
    uniform_order_statistic_medians(n)
        .iter()
        .map(|&m| ndtri(m))
        .collect()
}

/// `1 − r` where `r` is Pearson's correlation between `xvals` and the sorted
/// Box-Cox transform of `samps`. `xvals` must be ascending (it is, by
/// construction). Matches `scipy.stats.pearsonr`'s centered formulation.
pub fn eval_pearsonr(lmbda: f64, xvals: &[f64], samps: &[f64]) -> f64 {
    let mut yvals = boxcox(samps, lmbda);
    yvals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    1.0 - pearson_r(xvals, &yvals)
}

/// Pearson correlation as `scipy.stats.pearsonr` computes it: center, scale each
/// centered vector by its max-element L2 norm, then dot the unit vectors. The
/// per-vector normalization (rather than a raw covariance ratio) is what keeps
/// it bit-aligned with scipy near |r| = 1.
fn pearson_r(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let xmean = pairwise_sum(x) / n;
    let ymean = pairwise_sum(y) / n;

    let xm: Vec<f64> = x.iter().map(|&v| v - xmean).collect();
    let ym: Vec<f64> = y.iter().map(|&v| v - ymean).collect();

    let normxm = scaled_norm(&xm);
    let normym = scaled_norm(&ym);

    let prod: Vec<f64> = xm
        .iter()
        .zip(&ym)
        .map(|(&a, &b)| (a / normxm) * (b / normym))
        .collect();
    pairwise_sum(&prod).clamp(-1.0, 1.0)
}

/// `‖v‖₂` computed as `max * ‖v/max‖₂`, scipy's overflow-safe L2 norm.
fn scaled_norm(v: &[f64]) -> f64 {
    let vmax = v.iter().fold(0.0_f64, |m, &a| m.max(a.abs()));
    let sq: Vec<f64> = v.iter().map(|&a| (a / vmax) * (a / vmax)).collect();
    vmax * pairwise_sum(&sq).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn medians_match_filliben() {
        // scipy `_calc_uniform_order_statistic_medians(5)`.
        let v = uniform_order_statistic_medians(5);
        let want = [
            0.129_449_436_703_875_88,
            0.313_606_710_158_434_3,
            0.5,
            0.686_393_289_841_565_7,
            0.870_550_563_296_124_1,
        ];
        for (g, w) in v.iter().zip(&want) {
            assert!((g - w).abs() < 1e-15, "got {g} want {w}");
        }
    }

    #[test]
    fn perfect_correlation_is_zero_objective() {
        let x = [-1.0, 0.0, 1.0, 2.0];
        let y = [2.0, 4.0, 6.0, 8.0];
        assert!((pearson_r(&x, &y) - 1.0).abs() < 1e-12);
    }
}

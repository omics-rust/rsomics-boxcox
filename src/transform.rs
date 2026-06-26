//! Box-Cox transform and profile log-likelihood, value-exact to scipy 1.17.1.
//!
//! The transform is cephes `scipy.special.boxcox`: `expm1(λ·ln x)/λ` for λ≠0
//! and `ln x` for λ=0 — the `expm1` form is what keeps small-λ values accurate
//! and bit-identical to cephes. The log-likelihood follows scipy's `_boxcox_llf`
//! exactly, including its trick of computing the variance of the transformed
//! data in log-space (`logsumexp`) and factoring out the `1/λ` offset, which
//! the naive `var(boxcox(x))` form fails to reproduce near λ=0.

use crate::sum::pairwise_sum;

/// Cephes `scipy.special.boxcox` for a single value.
#[inline]
pub fn boxcox1(x: f64, lmbda: f64) -> f64 {
    if lmbda == 0.0 {
        x.ln()
    } else {
        (lmbda * x.ln()).exp_m1() / lmbda
    }
}

/// Transform every element of `x` with parameter `lmbda`.
pub fn boxcox(x: &[f64], lmbda: f64) -> Vec<f64> {
    if lmbda == 0.0 {
        x.iter().map(|&v| v.ln()).collect()
    } else {
        x.iter()
            .map(|&v| (lmbda * v.ln()).exp_m1() / lmbda)
            .collect()
    }
}

/// Box-Cox profile log-likelihood `boxcox_llf(lmb, x)`.
///
/// `l = (λ−1)·Σ ln xᵢ − N/2·ln(var(y))`, with `var(y)` the population variance
/// of the transformed data. For λ≠0 the variance is evaluated in log-space to
/// match scipy's numerically stable path.
pub fn boxcox_llf(lmb: f64, data: &[f64]) -> f64 {
    let n = data.len();
    if n == 0 {
        return f64::NAN;
    }
    let nf = n as f64;

    let logdata: Vec<f64> = data.iter().map(|&v| v.ln()).collect();
    let sum_logdata = pairwise_sum(&logdata);

    let logvar = if lmb == 0.0 {
        log_var_direct(&logdata, sum_logdata, nf)
    } else {
        let logx: Vec<f64> = logdata.iter().map(|&l| lmb * l).collect();
        log_var(&logx, nf) - 2.0 * lmb.abs().ln()
    };

    (lmb - 1.0) * sum_logdata - nf / 2.0 * logvar
}

/// `ln(var(x))` from `logdata = ln(x)`, used at λ=0 where `var(ln x)` is real.
fn log_var_direct(logdata: &[f64], sum_logdata: f64, nf: f64) -> f64 {
    let mean = sum_logdata / nf;
    let dev2: Vec<f64> = logdata.iter().map(|&l| (l - mean) * (l - mean)).collect();
    (pairwise_sum(&dev2) / nf).ln()
}

/// scipy `_log_var`: `ln(var(exp(logx)))` evaluated in log-space.
///
/// `logmean = logsumexp(logx) − ln N`; `logxmu = logsumexp([logx, logmean],
/// b=[1, −1])` element-wise; `ln var = logsumexp(2·logxmu) − ln N`.
fn log_var(logx: &[f64], nf: f64) -> f64 {
    let ln_n = nf.ln();
    let logmean = logsumexp(logx) - ln_n;

    let log_sq: Vec<f64> = logx
        .iter()
        .map(|&lx| 2.0 * logsumexp2_signed(lx, logmean))
        .collect();

    logsumexp(&log_sq) - ln_n
}

/// `logsumexp(a)` = `max + ln(Σ exp(aᵢ − max))`, reducing with numpy's pairwise
/// sum so the result matches `scipy.special.logsumexp` bit-for-bit.
fn logsumexp(a: &[f64]) -> f64 {
    let amax = a.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if amax == f64::NEG_INFINITY {
        return amax;
    }
    let exps: Vec<f64> = a.iter().map(|&v| (v - amax).exp()).collect();
    pairwise_sum(&exps).ln() + amax
}

/// `ln|exp(a) − exp(b)|` via the two-term signed logsumexp scipy uses for
/// `logx − logmean`. The sign is dropped because the result is squared.
fn logsumexp2_signed(a: f64, b: f64) -> f64 {
    let amax = a.max(b);
    let s = (a - amax).exp() - (b - amax).exp();
    s.abs().ln() + amax
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transform_lambda_zero_is_log() {
        let x = [1.0, 2.0, 4.0];
        let y = boxcox(&x, 0.0);
        for (yi, xi) in y.iter().zip(x.iter()) {
            assert_eq!(*yi, xi.ln());
        }
    }

    #[test]
    fn transform_lambda_one_shifts_by_one() {
        let x = [1.0, 2.0, 3.0];
        let y = boxcox(&x, 1.0);
        for (yi, xi) in y.iter().zip(x.iter()) {
            assert!((yi - (xi - 1.0)).abs() < 1e-12);
        }
    }

    #[test]
    fn boxcox1_matches_slice() {
        let x = [0.5, 2.0, 7.3, 100.0];
        for lmb in [0.0, 0.3, -0.7, 2.0] {
            let v = boxcox(&x, lmb);
            for (i, &xi) in x.iter().enumerate() {
                assert_eq!(v[i], boxcox1(xi, lmb));
            }
        }
    }
}

//! `boxcox_normmax` — the optimal Box-Cox lambda by MLE or Pearson correlation.
//!
//! Both methods minimize a scalar objective with `optimize.brent` over the
//! default bracket `(-2.0, 2.0)`. MLE minimizes `−boxcox_llf`; Pearson minimizes
//! `1 − r` of the probability plot. scipy then clamps the result so the
//! transform cannot overflow the input dtype; that guard only fires for
//! near-degenerate data driving lambda to extremes.

use rsomics_common::{Result, RsomicsError};

use crate::optimize::brent;
use crate::pearson::{eval_pearsonr, normal_quantiles};
use crate::transform::{boxcox_llf, boxcox1};

const BRACK_LO: f64 = -2.0;
const BRACK_HI: f64 = 2.0;

/// Which objective `boxcox_normmax` optimizes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormmaxMethod {
    Pearsonr,
    Mle,
}

impl NormmaxMethod {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "pearsonr" => Ok(Self::Pearsonr),
            "mle" => Ok(Self::Mle),
            other => Err(RsomicsError::InvalidInput(format!(
                "unknown --normmax-method '{other}' (expected pearsonr or mle)"
            ))),
        }
    }
}

/// Optimal lambda for `data` by the chosen method.
pub fn boxcox_normmax(data: &[f64], method: NormmaxMethod) -> Result<f64> {
    if !data.iter().all(|&v| v.is_finite() && v >= 0.0) {
        return Err(RsomicsError::InvalidInput(
            "boxcox_normmax requires only positive, finite, real numbers".into(),
        ));
    }

    let lmbda = match method {
        NormmaxMethod::Mle => brent(|lmb| -boxcox_llf(lmb, data), BRACK_LO, BRACK_HI),
        NormmaxMethod::Pearsonr => {
            let xvals = normal_quantiles(data.len());
            brent(|lmb| eval_pearsonr(lmb, &xvals, data), BRACK_LO, BRACK_HI)
        }
    };

    enforce_ymax(data, lmbda)?;
    Ok(lmbda)
}

/// scipy's overflow guard: the transformed extreme value must stay within the
/// input dtype's range. On real data this never fires; if it would, scipy
/// substitutes a constrained lambda — we instead fail loudly rather than emit a
/// silently different result.
fn enforce_ymax(data: &[f64], lmbda: f64) -> Result<()> {
    let ymax = f64::MAX / 10000.0;
    let xmax = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let xmin = data.iter().copied().fold(f64::INFINITY, f64::min);

    let x_treme = if xmin >= 1.0 {
        xmax
    } else if xmax <= 1.0 {
        xmin
    } else if boxcox1(xmax, lmbda) > boxcox1(xmin, lmbda).abs() {
        xmax
    } else {
        xmin
    };

    if boxcox1(x_treme, lmbda).abs() > ymax {
        return Err(RsomicsError::InvalidInput(format!(
            "optimal lambda {lmbda} would overflow the transform; data is too \
             near-degenerate for an unconstrained Box-Cox fit"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn method_parse() {
        assert_eq!(NormmaxMethod::parse("mle").unwrap(), NormmaxMethod::Mle);
        assert_eq!(
            NormmaxMethod::parse("pearsonr").unwrap(),
            NormmaxMethod::Pearsonr
        );
        assert!(NormmaxMethod::parse("nope").is_err());
    }

    #[test]
    fn rejects_nonpositive() {
        let data = [1.0, 2.0, -1.0];
        assert!(boxcox_normmax(&data, NormmaxMethod::Mle).is_err());
    }
}

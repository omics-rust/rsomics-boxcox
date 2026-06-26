//! Box-Cox power transform, profile log-likelihood, and optimal lambda —
//! value-exact to `scipy.stats.boxcox` / `boxcox_llf` / `boxcox_normmax`
//! (scipy 1.17.1).
//!
//! Given a strictly positive sample, `boxcox` either applies the transform for a
//! supplied lambda or finds the MLE lambda that maximizes the log-likelihood
//! (Brent over the default `(-2, 2)` bracket) and transforms with it. The
//! `pearsonr` and `mle` normmax objectives are both exposed.

mod ndtri;
mod normmax;
mod optimize;
mod pearson;
mod sum;
mod transform;

use std::path::Path;

use rsomics_common::{Result, RsomicsError};

pub use normmax::{NormmaxMethod, boxcox_normmax};
pub use transform::{boxcox, boxcox_llf, boxcox1};

/// Read a whitespace-separated numeric file with no per-line String allocation:
/// the whole file lands in one buffer, tokens are sliced in place, and each is
/// parsed with Lemire's algorithm (fast-float2).
pub fn parse_values(path: &Path) -> Result<Vec<f64>> {
    let bytes = std::fs::read(path).map_err(RsomicsError::Io)?;
    parse_bytes(&bytes, &path.display().to_string())
}

/// Parse whitespace-separated floats from an in-memory buffer.
pub fn parse_bytes(bytes: &[u8], source: &str) -> Result<Vec<f64>> {
    let mut out = Vec::new();
    for tok in bytes.split(|b| b.is_ascii_whitespace()) {
        if tok.is_empty() {
            continue;
        }
        let v: f64 = fast_float2::parse(tok).map_err(|_| {
            let s = String::from_utf8_lossy(tok);
            RsomicsError::InvalidInput(format!("'{s}' is not a number in {source}"))
        })?;
        out.push(v);
    }
    if out.is_empty() {
        return Err(RsomicsError::InvalidInput(format!("no data in {source}")));
    }
    Ok(out)
}

/// Validate that every value is strictly positive and the sample is non-constant,
/// matching `scipy.stats.boxcox`'s preconditions for the MLE path.
pub fn check_positive_nonconstant(data: &[f64]) -> Result<()> {
    if data.iter().any(|&v| v <= 0.0) {
        return Err(RsomicsError::InvalidInput("data must be positive".into()));
    }
    let first = data[0];
    if data.iter().all(|&v| v == first) {
        return Err(RsomicsError::InvalidInput(
            "data must not be constant".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_one_per_line() {
        let v = parse_bytes(b"1.0\n2.5\n3\n", "x").unwrap();
        assert_eq!(v, vec![1.0, 2.5, 3.0]);
    }

    #[test]
    fn parses_whitespace_mixed() {
        let v = parse_bytes(b"1 2\t3\n4", "x").unwrap();
        assert_eq!(v, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn rejects_nonpositive_for_mle() {
        assert!(check_positive_nonconstant(&[1.0, 0.0, 2.0]).is_err());
        assert!(check_positive_nonconstant(&[3.0, 3.0, 3.0]).is_err());
        assert!(check_positive_nonconstant(&[1.0, 2.0, 3.0]).is_ok());
    }
}

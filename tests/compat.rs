//! Differential compatibility against committed scipy 1.17.1 goldens.
//!
//! `tests/golden/*.tsv` are the inputs; `tests/golden/golden.json` holds scipy's
//! reference outputs, generated once with scipy 1.17.1 (BSD-3) and committed, so
//! this test needs no Python at run time. The transform and log-likelihood are
//! checked to 1e-12 (`x.powf(λ)` is not bit-portable across architectures); the
//! MLE/Pearson optimal lambda to 1e-6 (an iterative optimizer's path tolerance).

use std::path::PathBuf;

use rsomics_boxcox::{NormmaxMethod, boxcox, boxcox_llf, boxcox_normmax, parse_values};
use serde_json::Value;

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden")
}

fn load_golden() -> Value {
    let raw = std::fs::read_to_string(golden_dir().join("golden.json")).unwrap();
    serde_json::from_str(&raw).unwrap()
}

/// Goldens store floats as Python `repr` strings; parse via Rust's correctly
/// rounded `str::parse` (serde_json's `as_f64` can be a ULP off) so the
/// comparison is against scipy's exact bits.
fn parse_repr(v: &Value) -> f64 {
    v.as_str().unwrap().parse().unwrap()
}

fn parse_repr_array(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(parse_repr).collect()
}

fn relerr(a: f64, b: f64) -> f64 {
    if b == 0.0 {
        a.abs()
    } else {
        (a - b).abs() / b.abs()
    }
}

// The transform is value-exact to scipy, but `x.powf(lmb)` differs by ~1–2 ULP
// between aarch64 (where the golden was generated) and x86_64 libm, so the check
// is a tight relative tolerance rather than bit equality.
#[test]
fn transform_matches_scipy_for_fixed_lambda() {
    let golden = load_golden();
    for (name, rec) in golden.as_object().unwrap() {
        let data = parse_values(&golden_dir().join(format!("{name}.tsv"))).unwrap();
        let fixed = rec["transform_fixed"].as_object().unwrap();
        for (lmb_s, arr) in fixed {
            let lmb: f64 = lmb_s.parse().unwrap();
            let ours = boxcox(&data, lmb);
            let want = parse_repr_array(arr);
            assert_eq!(ours.len(), want.len());
            for (o, w) in ours.iter().zip(&want) {
                let rel = (*o - *w).abs() / w.abs().max(f64::MIN_POSITIVE);
                assert!(
                    rel <= 1e-12,
                    "{name} lambda={lmb}: {o} != {w} (rel {rel:e})"
                );
            }
        }
    }
}

#[test]
fn llf_matches_scipy() {
    let golden = load_golden();
    for (name, rec) in golden.as_object().unwrap() {
        let data = parse_values(&golden_dir().join(format!("{name}.tsv"))).unwrap();
        for (lmb_s, want) in rec["llf"].as_object().unwrap() {
            let lmb: f64 = lmb_s.parse().unwrap();
            let ours = boxcox_llf(lmb, &data);
            let w = parse_repr(want);
            assert!(
                relerr(ours, w) <= 1e-12,
                "{name} llf({lmb}): {ours} vs {w} relerr {}",
                relerr(ours, w)
            );
        }
    }
}

#[test]
fn mle_lambda_matches_scipy() {
    let golden = load_golden();
    for (name, rec) in golden.as_object().unwrap() {
        let data = parse_values(&golden_dir().join(format!("{name}.tsv"))).unwrap();
        let ours = boxcox_normmax(&data, NormmaxMethod::Mle).unwrap();
        let want = parse_repr(&rec["lmax_mle"]);
        assert!(
            (ours - want).abs() <= 1e-6,
            "{name} MLE lambda: {ours} vs {want}"
        );
    }
}

#[test]
fn pearsonr_lambda_matches_scipy() {
    let golden = load_golden();
    for (name, rec) in golden.as_object().unwrap() {
        let data = parse_values(&golden_dir().join(format!("{name}.tsv"))).unwrap();
        let ours = boxcox_normmax(&data, NormmaxMethod::Pearsonr).unwrap();
        let want = parse_repr(&rec["lmax_pearsonr"]);
        assert!(
            (ours - want).abs() <= 1e-6,
            "{name} Pearson lambda: {ours} vs {want}"
        );
    }
}

#[test]
fn transform_at_mle_lambda_matches_scipy() {
    let golden = load_golden();
    for (name, rec) in golden.as_object().unwrap() {
        let data = parse_values(&golden_dir().join(format!("{name}.tsv"))).unwrap();
        let lmax = boxcox_normmax(&data, NormmaxMethod::Mle).unwrap();
        let ours = boxcox(&data, lmax);
        let want = parse_repr_array(&rec["transform_mle"]);
        for (o, w) in ours.iter().zip(&want) {
            // Optimal-lambda differs by an optimizer-path epsilon, so the
            // transform tracks it rather than being bit-exact.
            assert!(
                relerr(*o, *w) <= 1e-6 || (o - w).abs() <= 1e-9,
                "{name} transform@MLE: {o} vs {w}"
            );
        }
    }
}

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use rsomics_boxcox::{NormmaxMethod, boxcox, boxcox_llf, boxcox_normmax};

fn sample(n: usize) -> Vec<f64> {
    // Deterministic skewed positive sample (gamma-ish via a simple recurrence).
    let mut x = Vec::with_capacity(n);
    let mut s: f64 = 0.123;
    for _ in 0..n {
        s = (s * 1103515245.0 + 12345.0).rem_euclid(2147483648.0);
        let u = s / 2147483648.0;
        x.push((-u.ln()) * 2.0 + 0.1);
    }
    x
}

fn bench(c: &mut Criterion) {
    let data = sample(200_000);

    c.bench_function("transform_lambda_0.3", |b| {
        b.iter(|| boxcox(black_box(&data), black_box(0.3)))
    });

    c.bench_function("llf", |b| {
        b.iter(|| boxcox_llf(black_box(0.3), black_box(&data)))
    });

    c.bench_function("normmax_mle", |b| {
        b.iter(|| boxcox_normmax(black_box(&data), NormmaxMethod::Mle).unwrap())
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);

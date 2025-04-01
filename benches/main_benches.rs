mod modules;

use criterion::{criterion_group, criterion_main, Criterion};
use crate::modules::classification::register_classification_benchmarks;
use crate::modules::regression::register_regression_benchmarks;

fn custom_criterion() -> Criterion {
    Criterion::default()
        // Increase allotted time to 100 seconds
        .measurement_time(std::time::Duration::from_secs(100))
        // Set amount of samples collected to 50
        .sample_size(50)
}
criterion_group! {
    name = benches;
    config = custom_criterion();
    targets =
        register_regression_benchmarks,
        register_classification_benchmarks
}
criterion_main!(benches);
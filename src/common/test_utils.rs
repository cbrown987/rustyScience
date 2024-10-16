#![cfg(test)]

use rand;
use rand::Rng;

pub(crate) fn generate_data(num_samples: usize, num_features: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut rng = rand::thread_rng();  // Create a random number generator

    // Generate random features
    let features: Vec<Vec<f64>> = (0..num_samples)
        .map(|_| {
            (0..num_features)
                .map(|_| rng.gen_range(0.0..100.0))  // Random floats between 0 and 100
                .collect()
        })
        .collect();

    // Generate random binary labels (0 or 1)
    let labels: Vec<f64> = (0..num_samples)
        .map(|_| rng.gen_range(0.0..2.0))  // Random integers (0 or 1)
        .collect();

    (features, labels)
}
// src/utils.rs

// Global utility functions


/// Calculates the Euclidean distance between two points.
pub(crate) fn euclidean_distance(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
}


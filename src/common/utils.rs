// src/utils.rs

// Global utility functions

pub(crate) fn euclidean_distance(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
}

pub(crate) fn manhattan_distance(x: &[f64], y: &[f64]) -> f64 {
    x.iter()
        .zip(y.iter())
        .map(|(&x, &y): (&f64, &f64)| (x - y).abs())
        .sum::<f64>()
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euclidean_distance() {
        // Test case 1: Same point, should return 0.0
        let point1 = vec![1.0, 2.0];
        let point2 = vec![1.0, 2.0];
        let result = euclidean_distance(&point1, &point2);
        assert!((result - 0.0).abs() < 1e-5, "Expected 0.0, got {}", result);

        // Test case 2: Simple points
        let point1 = vec![0.0, 0.0];
        let point2 = vec![3.0, 4.0];
        let result = euclidean_distance(&point1, &point2);
        assert!((result - 5.0).abs() < 1e-5, "Expected 5.0, got {}", result); // 3-4-5 triangle

        // Test case 3: Larger points
        let point1 = vec![1.0, 2.0, 3.0];
        let point2 = vec![4.0, 5.0, 6.0];
        let result = euclidean_distance(&point1, &point2);
        assert!((result - 5.19615242).abs() < 1e-5, "Expected 5.196, got {}", result);
    }

    #[test]
    fn test_manhattan_distance() {
        // Test case 1: Same point, should return 0.0
        let point1 = vec![1.0, 2.0];
        let point2 = vec![1.0, 2.0];
        let result = manhattan_distance(&point1, &point2);
        assert!((result - 0.0).abs() < 1e-5, "Expected 0.0, got {}", result);

        // Test case 2: Simple points
        let point1 = vec![0.0, 0.0];
        let point2 = vec![3.0, 4.0];
        let result = manhattan_distance(&point1, &point2);
        assert!((result - 7.0).abs() < 1e-5, "Expected 7.0, got {}", result); // 3 + 4

        // Test case 3: Larger points
        let point1 = vec![1.0, 2.0, 3.0];
        let point2 = vec![4.0, 5.0, 6.0];
        let result = manhattan_distance(&point1, &point2);
        assert!((result - 9.0).abs() < 1e-5, "Expected 9.0, got {}", result);

        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = manhattan_distance(&a, &b);
        let expected = 9.0; // Sum of |1-4| + |2-5| + |3-6|
        assert!((result - expected).abs() < 1e-6);
    }


    #[test]
    fn test_euclidean_zero_distance() {
        let a = vec![1.0, 1.0, 1.0];
        let b = vec![1.0, 1.0, 1.0];
        let result = euclidean_distance(&a, &b);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_manhattan_zero_distance() {
        let a = vec![1.0, 1.0, 1.0];
        let b = vec![1.0, 1.0, 1.0];
        let result = manhattan_distance(&a, &b);
        assert_eq!(result, 0.0);
    }
}


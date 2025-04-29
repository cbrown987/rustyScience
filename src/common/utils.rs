// src/utils.rs
use ndarray::Array2;
use ndarray_linalg::Inverse;
use num_traits::{Num, ToPrimitive};
// Global utility functions

pub(crate) fn euclidean_distance<D>(a: &[D], b: &[D]) -> f64
where
    D: Num + ToPrimitive + Copy,
{
    a.iter()
        .zip(b.iter())
        .map(|(x1, x2)| {
            let diff = x1.to_f64().unwrap() - x2.to_f64().unwrap();
            diff * diff
        })
        .sum::<f64>()
        .sqrt()
}


// fn invert_matrix(matrix: &Array2<f64>) -> Option<Array2<f64>> {
//     matrix.inv().ok()
// }

pub(crate) fn manhattan_distance<D>(a: &[D], b: &[D]) -> f64
where
    D: Num + ToPrimitive + Copy,
{
    a.iter()
        .zip(b.iter())
        .map(|(x1, x2)| {
            let diff = x1.to_f64().unwrap() - x2.to_f64().unwrap();
            diff.abs()
        })
        .sum()
}

pub(crate) fn shuffle_data_labels<D, L>(data: &mut Vec<Vec<D>>, labels: &mut Vec<L>) 
where
    L: Clone,
    D: Clone
{
    use rand::seq::SliceRandom;
    use rand::thread_rng;

    let mut indices: Vec<usize> = (0..data.len()).collect();
    indices.shuffle(&mut thread_rng());

    let shuffled_data: Vec<Vec<D>> = indices.iter().map(|&i| data[i].clone()).collect();
    let shuffled_labels: Vec<L> = indices.iter().map(|&i| labels[i].clone()).collect();

    *data = shuffled_data;
    *labels = shuffled_labels;
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


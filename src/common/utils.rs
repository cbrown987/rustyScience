// src/utils.rs
use ndarray::{Array1, Array2};
use ndarray_linalg::InverseC;
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


pub fn inverse_matrix(matrix: &Array2<f64>) -> Option<Array2<f64>> {
    matrix.invc().ok()
}

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

/// Calculate the dot product of two vectors
pub fn dot_product(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Matrix-vector multiplication
pub fn matrix_vector_mul(matrix: &Array2<f64>, vector: &Array1<f64>) -> Array1<f64> {
    let nrows = matrix.shape()[0];
    let mut result = Array1::<f64>::zeros(nrows);

    for i in 0..nrows {
        let row = matrix.slice(ndarray::s![i, ..]);
        result[i] = dot_product(&row.to_owned(), vector);
    }

    result
}

/// Solve a triangular system (simplified alternative to LAPACK's trtrs)
/// Only implements upper triangular solvers
pub fn solve_triangular(a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>, &'static str> {
    let n = a.shape()[0];
    if a.shape()[1] != n || b.len() != n {
        return Err("Dimension mismatch");
    }

    // Simple back substitution for upper triangular
    let mut x = Array1::<f64>::zeros(n);

    for i in (0..n).rev() {
        let mut sum = 0.0;
        for j in (i+1)..n {
            sum += a[[i, j]] * x[j];
        }

        if a[[i, i]] == 0.0 {
            return Err("Singular matrix");
        }

        x[i] = (b[i] - sum) / a[[i, i]];
    }

    Ok(x)
}

/// Simple Cholesky decomposition (alternative to LAPACK's potrf)
pub fn cholesky(a: &Array2<f64>) -> Result<Array2<f64>, &'static str> {
    let n = a.shape()[0];
    if a.shape()[1] != n {
        return Err("Matrix must be square");
    }

    let mut l = Array2::<f64>::zeros((n, n));

    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;

            if j == i {
                // Diagonal elements
                for k in 0..j {
                    sum += l[[j, k]] * l[[j, k]];
                }
                let val = a[[j, j]] - sum;
                if val <= 0.0 {
                    return Err("Matrix is not positive definite");
                }
                l[[j, j]] = val.sqrt();
            } else {
                // Off-diagonal elements
                for k in 0..j {
                    sum += l[[i, k]] * l[[j, k]];
                }
                if l[[j, j]] == 0.0 {
                    return Err("Division by zero");
                }
                l[[i, j]] = (a[[i, j]] - sum) / l[[j, j]];
            }
        }
    }

    Ok(l)
}

pub fn array2_to_vec(arr: &Array2<f64>) -> Vec<Vec<f64>> {
    let shape = arr.shape();
    let mut result = vec![vec![0.0; shape[1]]; shape[0]];
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            result[i][j] = arr[[i, j]];
        }
    }
    result
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


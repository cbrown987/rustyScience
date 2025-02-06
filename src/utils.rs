use ndarray::{Array2, linalg::Inverse};

pub fn inverse_matrix(matrix: &Array2<f64>) -> Option<Array2<f64>> {
    matrix.inv().ok()
}
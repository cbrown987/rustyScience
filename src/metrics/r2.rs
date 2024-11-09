use num::Num;
use std::convert::Into;

/// Calculates the R-squared (R²) score between two vectors of true values and predicted values.
///
/// # Type Parameters
/// - `T`: The type of the elements in the vectors. `T` must implement `Num`, `Copy`, and be convertible into `f64`.
///
/// # Parameters
/// - `y_true`: A vector of true values.
/// - `y_pred`: A vector of predicted values.
///
/// # Returns
/// - A `f64` value representing the R-squared score between the true and predicted values.
///
/// # Panics
/// - The function panics if the lengths of `y_true` and `y_pred` do not match.
/// - The function panics if `y_true` or `y_pred` are empty.
///
/// # Example
/// ```
/// use rusty_science::metrics::r2;
/// let y_true = vec![1.0, 2.0, 3.0, 4.0];
/// let y_pred = vec![0.8, 2.1, 2.9, 4.2];
/// let r2_score = r2(y_true, y_pred);
/// println!("R² Score: {}", r2_score);
/// ```
pub fn r2<T>(y_true: Vec<T>, y_pred: Vec<T>) -> f64
where
    T: Num + Into<f64> + Copy,
{
    if y_true.len() != y_pred.len() {
        panic!("Vectors y_true and y_pred must have the same length");
    }
    if y_true.len() == 0 {
        panic!("Vectors y_true and y_pred must have at least one element");
    }

    let mean_y_true: f64 = y_true.iter().map(|&val| val.into()).sum::<f64>() / y_true.len() as f64;

    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;

    for (y, y_hat) in y_true.iter().zip(y_pred.iter()) {
        let y: f64 = (*y).into();
        let y_hat: f64 = (*y_hat).into();
        ss_res += (y - y_hat).powi(2);
        ss_tot += (y - mean_y_true).powi(2);
    }

    1.0 - (ss_res / ss_tot)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_correlation() {
        let y_true = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_pred = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let r2 = r2(y_true, y_pred);
        assert!((r2 - 1.0).abs() < 1e-9, "Expected R^2 close to 1, got {}", r2);
    }

    #[test]
    fn test_no_correlation() {
        let y_true = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_pred = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let r2 = r2(y_true, y_pred);
        assert!((r2 - (-3.0)).abs() < 1e-9, "Expected R^2 close to -3, got {}", r2);
    }

    #[test]
    fn test_some_correlation() {
        let y_true = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_pred = vec![1.1, 1.9, 3.2, 3.9, 5.1];
        let r2 = r2(y_true, y_pred);
        assert!(r2 > 0.9, "Expected R^2 > 0.9, got {}", r2);
    }

    #[test]
    fn test_identical_values() {
        let y_true = vec![2.0, 2.0, 2.0, 2.0, 2.0];
        let y_pred = vec![2.0, 2.0, 2.0, 2.0, 2.0];
        let r2 = r2(y_true, y_pred);
        assert!(r2.is_nan(), "Expected R^2 to be NaN due to zero variance, got {}", r2);
    }

    #[test]
    #[should_panic]
    fn test_mismatched_lengths() {
        let y_true = vec![1.0, 2.0, 3.0];
        let y_pred = vec![1.0, 2.0];
        let _r2 = r2(y_true, y_pred);
    }

    #[test]
    #[should_panic]
    fn test_empty_vectors() {
        let y_true: Vec<f64> = vec![];
        let y_pred: Vec<f64> = vec![];
        let _r2 = r2(y_true, y_pred);
    }
}




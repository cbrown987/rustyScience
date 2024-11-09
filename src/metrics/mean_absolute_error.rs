/// Calculates the Mean Absolute Error (MAE) between two vectors of true values and predicted values.
///
/// # Type Parameters
/// - `T`: The type of the elements in the vectors. `T` must implement `Copy` and be convertible into `f64`.
///
/// # Parameters
/// - `y_true`: A vector of true values.
/// - `y_pred`: A vector of predicted values.
///
/// # Returns
/// - A `f64` value representing the mean absolute error between the true and predicted values.
///
/// # Panics
/// - The function panics if the lengths of `y_true` and `y_pred` do not match.
/// - The function panics if `y_true` or `y_pred` are empty.
///
/// # Example
/// ```
/// use rusty_science::metrics::mean_absolute_error;
/// let y_true = vec![1.0, 2.0, 3.0];
/// let y_pred = vec![1.5, 2.5, 3.0];
/// let mae = mean_absolute_error(y_true, y_pred);
/// println!("Mean Absolute Error: {}", mae);
/// ```
pub fn mean_absolute_error<T: Copy + Into<f64>, >(y_true: Vec<T>, y_pred: Vec<T>) -> f64 {
    if y_true.len() != y_pred.len() {
        panic!("Vectors y_true and y_pred must have the same length")
    }
    if y_true.len() == 0 {
        panic!("Vectors y_true and y_pred must have at least one element");
    }

    let sum_absolute_errors: f64 = y_true.iter()
        .zip(y_pred.iter())
        .map(|(&a, &p)| (a.into() - p.into()).abs())
        .sum();

    sum_absolute_errors / y_true.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean_absolute_error_basic() {
        let y_true = vec![1.0, 2.0, 3.0];
        let y_pred = vec![1.0, 2.0, 2.0];
        let mae = mean_absolute_error(y_true, y_pred);
        assert!((mae - 0.333333).abs() < 1e-6);
    }

    #[test]
    fn test_mean_absolute_error_integers() {
        let y_true = vec![1, 2, 3, 4];
        let y_pred = vec![2, 2, 4, 3];
        let mae = mean_absolute_error(y_true, y_pred);
        assert_eq!(mae, 0.75);
    }

    #[test]
    fn test_mean_absolute_error_zero_error() {
        let y_true = vec![5.0, 6.0, 7.0, 8.0];
        let y_pred = vec![5.0, 6.0, 7.0, 8.0];
        let mae = mean_absolute_error(y_true, y_pred);
        assert_eq!(mae, 0.0);
    }


    #[test]
    #[should_panic]
    fn test_mean_absolute_error_empty_vectors() {
        let y_true: Vec<f64> = vec![];
        let y_pred: Vec<f64> = vec![];
        mean_absolute_error(y_true, y_pred);
    }

    #[test]
    #[should_panic]
    fn test_mean_absolute_error_mismatched_lengths() {
        let y_true = vec![1.0, 2.0, 3.0];
        let y_pred = vec![1.0, 2.0];
        mean_absolute_error(y_true, y_pred);
    }

    #[test]
    fn test_mean_absolute_error_large_values() {
        let y_true = vec![1e10, 2e10, 3e10];
        let y_pred = vec![1.1e10, 1.9e10, 3.1e10];
        let mae = mean_absolute_error(y_true, y_pred);
        assert!((mae - 1e9).abs() < 1e-6);
    }

    #[test]
    fn test_mean_absolute_error_negative_values() {
        let y_true = vec![-1.0, -2.0, -3.0];
        let y_pred = vec![-1.5, -2.5, -3.5];
        let mae = mean_absolute_error(y_true, y_pred);
        assert!((mae - 0.5).abs() < 1e-6);
    }
}

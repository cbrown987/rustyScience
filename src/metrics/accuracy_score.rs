use num::Num;

/// Calculates the normalized accuracy score between two vectors of true values and predicted values.
///
/// # Type Parameters
/// - `T`: The type of the elements in the vectors. `T` must implement `Num`.
///
/// # Parameters
/// - `data_true`: A vector of true values.
/// - `data_pred`: A vector of predicted values.
///
/// # Returns
/// - A `f64` value representing the normalized accuracy score between the true and predicted values.
///
/// # Example
/// ```
/// use rusty_science::metrics::accuracy_score_normalize;
/// let data_true = vec![1, 2, 3, 4, 5];
/// let data_pred = vec![1, 2, 2, 4, 5];
/// let accuracy = accuracy_score_normalize(data_true, data_pred);
/// println!("Normalized Accuracy Score: {}", accuracy);
/// ```
pub fn accuracy_score_normalize<T: Num>(data_true: Vec<T>, data_pred: Vec<T>) -> f64 {
    let min_len = data_true.len().min(data_pred.len());
    let matches = data_true.iter()
        .zip(data_pred.iter())
        .take(min_len)
        .filter(|(a, b)| a == b)
        .count();

    let total_len = data_true.len().max(data_pred.len()) as f64;
    matches as f64 / total_len
}


/// Calculates the accuracy score between two vectors of true values and predicted values.
///
/// # Type Parameters
/// - `T`: The type of the elements in the vectors. `T` must implement `Num`.
///
/// # Parameters
/// - `data_true`: A vector of true values.
/// - `data_pred`: A vector of predicted values.
///
/// # Returns
/// - An `i64` value representing the count of matching elements between the true and predicted values.
///
/// # Example
/// ```
/// use rusty_science::metrics::accuracy_score;
/// let data_true = vec![1, 2, 3, 4, 5];
/// let data_pred = vec![1, 2, 2, 4, 5];
/// let accuracy = accuracy_score(data_true, data_pred);
/// println!("Accuracy Score: {}", accuracy);
/// ```
pub fn accuracy_score<T: Num>(data_true: Vec<T>, data_pred: Vec<T>) -> i64 {
    let min_len = data_true.len().min(data_pred.len());
    let matches = data_true.iter()
        .zip(data_pred.iter())
        .take(min_len)
        .filter(|(a, b)| a == b)
        .count();

    matches as i64
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accuracy_score_normalize() {
        let data_test_one = vec![1, 2, 3, 4, 5, 6, 7];
        let data_pred_one = vec![1, 2, 3, 4, 5, 6, 7];
        let accuracy = accuracy_score_normalize(data_test_one, data_pred_one);
        assert_eq!(accuracy, 1.0);

        let data_test_two = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let data_pred_two = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 11];
        let accuracy = accuracy_score_normalize(data_test_two, data_pred_two);
        assert_eq!(accuracy, 0.9);

        let data_test_three = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let data_pred_three = vec![1, 2, 3, 4, 5, 6, 7];
        let accuracy = accuracy_score_normalize(data_test_three, data_pred_three);
        assert_eq!(accuracy, 0.7);

        let data_test_four = vec![1, 2, 3, 4, 5, 6, 7];
        let data_pred_four = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let accuracy = accuracy_score_normalize(data_test_four, data_pred_four);
        assert_eq!(accuracy, 0.7);

        let data_test_four = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let data_pred_four = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let accuracy = accuracy_score_normalize(data_test_four, data_pred_four);
        assert_eq!(accuracy, 0.7);
    }

    #[test]
    fn test_accuracy_score() {
        let data_test_one = vec![1, 2, 3, 4, 5, 6, 7];
        let data_pred_one = vec![1, 2, 3, 4, 5, 6, 7];
        let accuracy = accuracy_score(data_test_one, data_pred_one);
        assert_eq!(accuracy, 7);

        let data_test_two = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let data_pred_two = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 11];
        let accuracy = accuracy_score(data_test_two, data_pred_two);
        assert_eq!(accuracy, 9);

        let data_test_three = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let data_pred_three = vec![1, 2, 3, 4, 5, 6, 7];
        let accuracy = accuracy_score(data_test_three, data_pred_three);
        assert_eq!(accuracy, 7);

        let data_test_four = vec![1, 2, 3, 4, 5, 6, 7];
        let data_pred_four = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let accuracy = accuracy_score(data_test_four, data_pred_four);
        assert_eq!(accuracy, 7);
    }
}


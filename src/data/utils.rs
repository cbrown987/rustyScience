use rand::seq::SliceRandom;
use rand::thread_rng;

/// Splits the given dataset and labels into training and testing subsets.
///
/// # Arguments
/// * `data` - A vector of feature vectors (each feature vector being a vector of f64).
/// * `labels` - A vector of labels corresponding to the feature vectors.
/// * `train_ratio` - A floating point value representing the ratio of data that should go into the training set (e.g., 0.8 for an 80-20 split).
///
/// # Returns
/// A tuple containing two tuples:
/// 1. Training data and labels.
/// 2. Testing data and labels.
///
/// # Panics
/// The function will panic if `data` and `labels` have different lengths.
///
/// # Example
/// ```
///
/// use rusty_science::data::utils::train_test_split;
/// let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0], vec![7.0, 8.0]];
/// let labels = vec![0, 1, 0, 1];
/// let train_ratio = 0.75;
/// let ((train_data, train_labels), (test_data, test_labels)) = train_test_split(data, labels, train_ratio);
/// ```


pub fn train_test_split(data: Vec<Vec<f64>>, labels: Vec<i64>, train_ratio: f64)
                        -> ((Vec<Vec<f64>>, Vec<i64>), (Vec<Vec<f64>>, Vec<i64>)) {

    assert_eq!(data.len(), labels.len(), "Data and labels must have the same length");

    let mut combined: Vec<(Vec<f64>, i64)> = data.into_iter().zip(labels.into_iter()).collect();

    let mut rng = thread_rng();
    combined.shuffle(&mut rng);

    let train_size = (combined.len() as f64 * train_ratio).round() as usize;

    let (train_set, test_set) = combined.split_at(train_size);
    let (train_data, train_labels): (Vec<_>, Vec<_>) = train_set.iter().cloned().unzip();
    let (test_data, test_labels): (Vec<_>, Vec<_>) = test_set.iter().cloned().unzip();

    ((train_data, train_labels), (test_data, test_labels))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_train_test_split_ratio() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0], vec![7.0, 8.0]];
        let labels = vec![0, 1, 0, 1];
        let train_ratio = 0.75;

        let ((train_data, train_labels), (test_data, test_labels)) = train_test_split(data, labels, train_ratio);

        assert_eq!(train_data.len(), 3);
        assert_eq!(train_labels.len(), 3);
        assert_eq!(test_data.len(), 1);
        assert_eq!(test_labels.len(), 1);
    }

    #[test]
    fn test_train_test_split_all_train() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let labels = vec![0, 1];
        let train_ratio = 1.0;

        let ((train_data, train_labels), (test_data, test_labels)) = train_test_split(data, labels, train_ratio);

        assert_eq!(train_data.len(), 2);
        assert_eq!(train_labels.len(), 2);
        assert_eq!(test_data.len(), 0);
        assert_eq!(test_labels.len(), 0);
    }

    #[test]
    fn test_train_test_split_all_test() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let labels = vec![0, 1];
        let train_ratio = 0.0;

        let ((train_data, train_labels), (test_data, test_labels)) = train_test_split(data, labels, train_ratio);

        assert_eq!(train_data.len(), 0);
        assert_eq!(train_labels.len(), 0);
        assert_eq!(test_data.len(), 2);
        assert_eq!(test_labels.len(), 2);
    }

    #[test]
    #[should_panic(expected = "Data and labels must have the same length")]
    fn test_train_test_split_mismatched_lengths() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let labels = vec![0];
        let train_ratio = 0.5;

        train_test_split(data, labels, train_ratio);
    }
}

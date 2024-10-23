#![cfg(test)]

use std::collections::HashMap;

pub(crate) fn create_data_unlabeled () -> HashMap<String, Vec<Vec<f64>>> {
    let small_data: Vec<Vec<f64>> = vec![
        vec![0.0, 0.0],
        vec![1.0, 1.0],
        vec![2.0, 2.0]
    ];

    let mut datasets = HashMap::new();

    // Insert datasets with names
    datasets.insert("small_data".to_string(), small_data);

    datasets
}
pub(crate) struct LabeledDataset {
    pub(crate) data: Vec<Vec<f64>>,
    pub(crate) labels: Vec<i64>
}
pub(crate) fn create_data_labeled () -> HashMap<String, LabeledDataset> {
    let small_data: Vec<Vec<f64>> = vec![
        vec![0.0, 0.0],
        vec![1.0, 1.0],
        vec![2.0, 2.0]
    ];
    let small_labels: Vec<i64> = vec![
        1,
        1,
        10
    ];
    let small_data_joined = LabeledDataset { data: small_data, labels: small_labels };

    let mut datasets = HashMap::new();

    // Insert datasets with names
    datasets.insert("small_data".to_string(), small_data_joined);

    datasets
}
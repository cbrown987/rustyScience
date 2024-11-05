#![cfg(test)]

use std::io::Write;
use std::collections::HashMap;
use std::fs::File;

pub(crate) fn create_data_unlabeled() -> HashMap<String, Vec<Vec<f64>>> {
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
    pub(crate) labels: Vec<i64>,
}
pub(crate) fn create_data_labeled() -> HashMap<String, LabeledDataset> {
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

pub(crate) fn create_temp_csv(content: &str) -> String {
    let temp_dir = std::env::temp_dir();
    let file_path = temp_dir.join("_test_data.csv");
    let mut file = File::create(&file_path).expect("Unable to create test file");
    writeln!(file, "{}", content).expect("Unable to write to test file");
    file_path.to_string_lossy().into_owned()
}
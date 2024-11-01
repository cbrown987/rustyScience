use rustyScience::classification::knn::KNNClassifier;
use crate::data::test_import_data::{load_iris_data_labeled};

#[test]
pub(crate) fn test_knn_classification_integration() {
    let (data, labels) = load_iris_data_labeled();

    let target = vec![1.5, 1.5];

    let n_neighbors = 3;
    let mut knn = KNNClassifier::new( n_neighbors);
    knn.fit(data, labels);
    let prediction = knn.predict(target);

    assert_eq!(prediction, 1);
}

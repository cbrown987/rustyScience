use rustyScience::classification::knn::KNNClassifier;
use rustyScience::data::datasets::load_iris;

#[test]
pub(crate) fn test_knn_classification_integration() {
    let iris_data = load_iris();
    let (data, labels) = iris_data.to_numerical_labels();
    
    let target = vec![1.5, 1.5];

    let n_neighbors = 3;
    let mut knn = KNNClassifier::new(n_neighbors);
    knn.fit(data, labels);
    let prediction = knn.predict(target);

    assert_eq!(prediction, 0);
}

use rustyScience::classification::knn::KNNClassifier;

#[test]
fn test_knn_classification_integration() {
    let dataset = vec![
        vec![0.0, 0.0], // Index 0
        vec![1.0, 1.0], // Index 1
        vec![2.0, 2.0], // Index 2
        vec![3.0, 3.0], // Index 3
    ];
    let labels = vec![
        1,
        1,
        1,
        10
    ];

    // Define the target point
    let target = vec![1.5, 1.5];

    // Get the 2 nearest neighbors
    let n_neighbors = 3;
    let mut knn = KNNClassifier::new( n_neighbors);
    knn.fit(dataset, labels);
    let prediction = knn.predict(target);

    assert_eq!(prediction, 1);
}
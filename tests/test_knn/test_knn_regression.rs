use rusty_science::regression::knn::KNNRegression;
#[test]
pub(crate) fn test_knn_regression_integration(){
    let dataset = vec![
        vec![0.0, 0.0],
        vec![1.0, 1.0],
        vec![2.0, 2.0],
        vec![3.0, 3.0],
    ];
    let labels = vec![
        1.0,
        1.0,
        1.0,
        5.0
    ];

    let target = vec![1.5, 1.5];

    let n_neighbors = 3;
    let mut knn = KNNRegression::new( n_neighbors);
    knn.fit(dataset, labels);
    let prediction = knn.predict(target);

    assert_eq!(prediction, 1.0);
}
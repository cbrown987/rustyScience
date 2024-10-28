use rustyScience::clustering::knn::KNNCluster;

#[test]
fn test_knn_classification_integration() {
    let dataset = vec![
        vec![0.0, 0.0], // Index 0
        vec![1.0, 1.0], // Index 1
        vec![2.0, 2.0], // Index 2
        vec![3.0, 3.0], // Index 3
    ];
    
    let n_clusters = 2;
    let mut knn = KNNCluster::new(n_clusters);
    let clusters = knn.fit(dataset.clone());
    assert_eq!(clusters.len(), dataset.len());
    assert_eq!(clusters, vec![0,0,1,1]);
    
}
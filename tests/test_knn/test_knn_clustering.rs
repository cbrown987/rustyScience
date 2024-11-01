use std::collections::HashMap;
use rustyScience::clustering::knn::KNNCluster;
use crate::data::test_import_data::load_iris_data_labeled;


fn _evaluate_accuracy(predicted_clusters: Vec<usize>, actual_labels: Vec<i64>, cluster_label_map: &HashMap<usize, i64>) -> f64 {
    let correct = predicted_clusters
        .iter()
        .zip(actual_labels)
        .filter(|(&pred, actual)| cluster_label_map.get(&(pred as usize)) == Some(&actual))
        .count();

    correct as f64 / predicted_clusters.len() as f64
}

#[test]
pub(crate) fn test_knn_clustering_integration() {
    let (data, labels) = load_iris_data_labeled();
    
    let n_clusters = 3;
    let mut knn = KNNCluster::new(n_clusters);
    let predicted_clusters = knn.fit(data);
    let clusters_labeled = knn.map_cluster_to_label(labels.clone());
    let accuracy = _evaluate_accuracy(predicted_clusters, labels, &clusters_labeled);
    assert!(accuracy > 0.6);
}
use rusty_science::clustering::kmeans::KMeans;
use rusty_science::clustering::dbscan::Dbscan;
use rusty_science::data::datasets::{load_iris};

pub fn run_kmeans_demo() {
    println!("KMeans Clustering Demo");

    let iris_data = load_iris();
    let (data, labels) = iris_data.to_numerical_labels();


    let n_clusters = 3;
    let mut knn = KMeansCluster::new(n_clusters);
    let predicted_clusters = knn.fit(data);
    let clusters_labeled = knn.map_cluster_to_label(labels.clone());
    let accuracy = _evaluate_accuracy(predicted_clusters, labels, &clusters_labeled);
    assert!(accuracy > 0.6);
}
// pub fn run_dbscan_demo() {
// }

fn main() {
    println!("Running Clustering Models Demo...\n");


}
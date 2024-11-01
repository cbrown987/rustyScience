use std::collections::HashMap;
use crate::common::utils::{euclidean_distance, manhattan_distance};

pub struct KNNCluster {
    k: usize,
    data: Vec<Vec<f64>>,
    distance_metric: String,
    clusters: Vec<usize>,
}

fn _calculate_distance(a: &Vec<f64>, b: &Vec<f64>, distance_metric: String) -> f64 {
    match distance_metric.to_lowercase().as_str() {
        "euclidean" => { euclidean_distance(a, b) }
        "manhattan" => { manhattan_distance(a, b) }
        _ => panic!("Unsupported distance metric"),
    }
}

impl KNNCluster {
    /// Creates a new `KNNCluster` instance with a specified number of clusters (`k`).
    ///
    /// # Parameters
    /// - `k`: The number of clusters to form.
    ///
    /// # Returns
    /// - A new instance of `KNNCluster`.
    ///
    /// # Panics
    /// - The function panics if `k` is zero or less.
    ///
    /// # Example
    /// ```
    /// use rustyScience::clustering::knn::KNNCluster;
    /// let knn_cluster = KNNCluster::new(3);
    pub fn new(k: usize) -> Self {
        if k <= 0 {
            panic!("K cannot be zero");
        }
        Self {
            k,
            data: vec![vec![]],
            distance_metric: "euclidean".to_string(),
            clusters: vec![]
        }
    }

    /// Sets the distance metric for the KNN clustering algorithm.
    ///
    /// # Parameters
    /// - `distance_metric`: A string representing the type of distance metric to be used (e.g., "euclidean").
    ///
    /// # Example
    /// ```
    /// use rustyScience::clustering::knn::KNNCluster;
    /// let mut knn_cluster = KNNCluster::new(3);
    /// knn_cluster.set_distance_metrics("manhattan".to_string());
    /// ```
    pub fn set_distance_metrics(&mut self, distance_metric: String) {
        self.distance_metric = distance_metric;
    }

    /// Fits the KNN model to the provided data and returns the labels of the clusters.
    ///
    /// # Parameters
    /// - `data`: A vector of data points where each data point is represented as a vector of `f64` values.
    ///
    /// # Returns
    /// - A vector of `usize` values representing the cluster labels for each data point.
    ///
    /// # Example
    /// ```
    /// use rustyScience::clustering::knn::KNNCluster;
    /// let mut knn_cluster = KNNCluster::new(3);
    /// let data = vec![vec![1.0, 2.0], vec![1.5, 1.8], vec![5.0, 8.0]];
    /// let labels = knn_cluster.fit(data);
    /// println!("Cluster Labels: {:?}", labels);
    /// ```
    pub fn fit(&mut self, data: Vec<Vec<f64>>) -> Vec<usize> {
        self._fit(data)
    }

    fn _fit(&mut self, data: Vec<Vec<f64>>) -> Vec<usize> {
        self.data = data;

        let mut centroids: Vec<Vec<f64>> = self.data.iter().take(self.k).cloned().collect();
        let mut labels: Vec<usize> = vec![0; self.data.len()];

        let mut changed = true;
        while changed {
            changed = false;
            // Assign each point to the nearest centroid
            for (idx, point) in self.data.iter().enumerate() {
                let mut nearest_centroid = 0;
                let mut nearest_distance = f64::MAX;

                for (centroid_idx, centroid) in centroids.iter().enumerate() {
                    let distance = _calculate_distance(point, centroid, self.distance_metric.clone());
                    if distance < nearest_distance {
                        nearest_distance = distance;
                        nearest_centroid = centroid_idx;
                    }
                }

                if labels[idx] != nearest_centroid {
                    labels[idx] = nearest_centroid;
                    changed = true;
                }
            }

            // Update centroids by averaging assigned points
            let mut new_centroids: Vec<Vec<f64>> = vec![vec![0.0; centroids[0].len()]; self.k];
            let mut counts = vec![0; self.k];

            for (idx, point) in self.data.iter().enumerate() {
                let label = labels[idx];
                for (dim, value) in point.iter().enumerate() {
                    new_centroids[label][dim] += value;
                }
                counts[label] += 1;
            }

            for (centroid_idx, centroid) in new_centroids.iter_mut().enumerate() {
                if counts[centroid_idx] != 0 {
                    for value in centroid.iter_mut() {
                        *value /= counts[centroid_idx] as f64;
                    }
                }
            }

            centroids = new_centroids;
        }
        self.clusters = labels.clone();
        labels
    }

    pub fn map_cluster_to_label(&self, labels: Vec<i64>) -> HashMap<usize, i64> {
        let mut label_map: HashMap<usize, i64> = HashMap::new();
        let mut cluster_label_counts: HashMap<usize, HashMap<i64, usize>> = HashMap::new();

        // Iterate over each data point and update label counts per cluster
        for (idx, &cluster_id) in self.clusters.iter().enumerate() {
            let label = labels[idx];
            let label_counts = cluster_label_counts.entry(cluster_id).or_insert_with(HashMap::new);
            *label_counts.entry(label).or_insert(0) += 1;
        }

        // Determine the most frequent label for each cluster
        for (cluster_id, counts) in cluster_label_counts.iter() {
            if let Some((&most_common_label, _)) = counts.iter().max_by_key(|&(_, count)| count) {
                label_map.insert(*cluster_id, most_common_label);
            }
        }

        label_map
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cluster_assignment() {
        let mut clusterer = KNNCluster::new(2);
        let data = vec![
            vec![1.0, 2.0],
            vec![1.5, 1.8],
            vec![5.0, 8.0],
            vec![8.0, 8.0],
            vec![1.0, 0.6],
            vec![9.0, 11.0],
        ];
        let clusters = clusterer.fit(data);
        assert_eq!(clusters.len(), 6);
    }

    #[test]
    fn test_single_cluster() {
        let mut clusterer = KNNCluster::new(1);
        let data = vec![
            vec![1.0, 2.0],
            vec![1.5, 1.8],
            vec![5.0, 8.0],
        ];
        let clusters = clusterer.fit(data);
        assert!(clusters.iter().all(|&x| x == 0));
    }

    #[test]
    fn test_distance_metric() {
        let data = vec![
            vec![0.0, 0.0],
            vec![3.0, 4.0],
        ];
        let distance = _calculate_distance(&data[0], &data[1], "euclidean".to_string());
        assert_eq!(distance, 5.0);
    }

    #[test]
    fn test_map_cluster_to_label() {
        let mut knn_cluster = KNNCluster::new(2);

        // Mock data points and labels (simulating a result of clustering)
        let data = vec![
            vec![1.0, 2.0], // Cluster 0
            vec![1.5, 1.8], // Cluster 0
            vec![5.0, 8.0], // Cluster 1
            vec![6.0, 7.5]  // Cluster 1
        ];

        knn_cluster.fit(data);


        let labels = vec![1, 1, 2, 2]; // Points 0 and 1 get label 1; points 2 and 3 get label 2

        let label_map = knn_cluster.map_cluster_to_label(labels);

        // Expected mapping: Cluster 0 -> Label 1, Cluster 1 -> Label 2
        let mut expected_map = HashMap::new();
        expected_map.insert(0, 1);
        expected_map.insert(1, 2);

        // Assert the label map is as expected
        assert_eq!(label_map, expected_map);
    }
}
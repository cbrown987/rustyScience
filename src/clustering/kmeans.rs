//! # K-means Clustering Algorithm
//!
//! ## Theoretical Background
//!
//! K-means is one of the simplest and most widely used unsupervised machine learning algorithms for clustering. It:
//!
//! - Partitions data into k distinct, non-overlapping clusters
//! - Minimizes the within-cluster sum of squares (inertia)
//! - Creates clusters of approximately similar sizes and shapes (typically convex)
//! - Requires specifying the number of clusters (k) beforehand
//!
//! The algorithm works by iteratively assigning data points to the nearest cluster centroid and then updating those centroids based on the mean of the assigned points.
//!
//! ## Usage Examples
//!
//! Basic clustering with K-means:
//!
//! ```rust
//! use rusty_science::clustering::KMeansCluster;
//!
//! // Create example data
//! let data = vec![
//!     vec![1.0, 2.0], vec![1.1, 2.2], vec![0.9, 1.9], vec![1.0, 2.1],  // Cluster 1
//!     vec![4.0, 5.0], vec![4.2, 5.1], vec![3.9, 4.8], vec![4.1, 5.2],  // Cluster 2
//!     vec![10.0, 10.0]  // Outlier
//! ];
//!
//! // Create and configure K-means
//! let mut kmeans = KMeansCluster::new(2);  // 2 clusters
//! kmeans.set_distance_metrics("euclidean".to_string());  // Use Euclidean distance
//!
//! // Fit the model
//! let cluster_labels = kmeans.fit(data.clone());
//! println!("Cluster labels: {:?}", cluster_labels);
//!
//! // Map labels to clusters (if you have predefined labels)
//! let original_labels = vec!["A", "A", "A", "A", "B", "B", "B", "B", "C"];
//! let cluster_map = kmeans.map_cluster_to_label(original_labels);
//! println!("Cluster to label mapping: {:?}", cluster_map);
//! ```
//!
//! ## Performance Characteristics
//!
//! - **Time Complexity**: O(n * k * d * i), where:
//!   - n is the number of points
//!   - k is the number of clusters
//!   - d is the dimensionality of the data
//!   - i is the number of iterations
//!
//! - **Space Complexity**: O(n + k), for storing point assignments and centroids
//!
//! - **Strengths**:
//!   - Simple implementation and intuitive understanding
//!   - Scales well to large datasets
//!   - Guarantees convergence (though potentially to local optima)
//!   - Works well with spherical or similarly sized clusters
//!
//! - **Weaknesses**:
//!   - Requires specifying k in advance
//!   - Sensitive to initial centroid selection
//!   - Struggles with non-spherical clusters
//!   - Not robust to outliers (they can significantly affect centroid positions)
//!   - May converge to local optima rather than global
//!

use std::collections::HashMap;
use std::hash::Hash;
use num_traits::{FromPrimitive, Num, ToPrimitive};
use crate::common::utils::{euclidean_distance, manhattan_distance};

pub struct KMeansCluster<D> {
    k: usize,
    data: Vec<Vec<D>>,
    distance_metric: String,
    clusters: Vec<usize>,
}


fn _calculate_distance<D>(a: &Vec<D>, b: &Vec<D>, distance_metric: &str) -> f64
where
    D: ToPrimitive + Copy + Num,
{
    match distance_metric.to_lowercase().as_str() {
        "euclidean" => euclidean_distance(a, b),
        "manhattan" => manhattan_distance(a, b),
        _ => panic!("Unsupported distance metric"),
    }
}

impl<D> KMeansCluster<D>
where
    D: Num + Copy + Clone + PartialOrd + ToPrimitive + FromPrimitive,
{
    /// Creates a new KMeansCluster with a specified value of k.
    ///
    /// # Arguments
    /// * `k` - The number of neighbors to consider for clustering. Must be greater than zero.
    ///
    /// # Panics
    /// This function will panic if `k` is less than or equal to zero.
    ///
    /// # Examples
    /// ```
    /// use rusty_science::clustering::KMeansCluster;
    /// let knn = KMeansCluster::<f64>::new(3);
    /// ```
    pub fn new(k: usize) -> Self {
        if k == 0 {
            panic!("K cannot be zero");
        }
        Self {
            k,
            data: vec![],
            distance_metric: "euclidean".to_string(),
            clusters: vec![],
        }
    }

    /// Sets the distance metric to be used for finding neighbors.
    ///
    /// # Arguments
    /// * `distance_metric` - A string specifying the distance metric, e.g., 'euclidean' or 'manhattan'.
    ///
    /// # Examples
    /// ```
    /// use rusty_science::clustering::KMeansCluster;
    /// let mut knn = KMeansCluster::<f64>::new(3);
    /// knn.set_distance_metrics("manhattan".to_string());
    /// ```
    pub fn set_distance_metrics(&mut self, distance_metric: String) {
        self.distance_metric = distance_metric;
    }
    
    /// Creates the clusters with the data provided to the Kmeans clusterer
    ///
    /// # Arguments
    /// * `data` - A 2d Vector of numbers representing the data
    ///
    /// # Examples
    /// ```
    /// use rusty_science::clustering::KMeansCluster;
    /// let mut knn = KMeansCluster::<f64>::new(3);
    /// let data = vec![vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 4.0]];
    /// let clusters = knn.fit(data);
    /// ```
    pub fn fit(&mut self, data: Vec<Vec<D>>) -> Vec<usize> {
        self._fit(data)
    }

    fn _fit(&mut self, data: Vec<Vec<D>>) -> Vec<usize> {
        self.data = data;

        let mut centroids: Vec<Vec<D>> = self.data.iter().take(self.k).cloned().collect();
        let mut labels: Vec<usize> = vec![0; self.data.len()];

        let mut changed = true;
        while changed {
            changed = false;
            // Assign each point to the nearest centroid
            for (idx, point) in self.data.iter().enumerate() {
                let mut nearest_centroid = 0;
                let mut nearest_distance = f64::MAX;

                for (centroid_idx, centroid) in centroids.iter().enumerate() {
                    let distance = _calculate_distance(point, centroid, &self.distance_metric);
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
            let dims = centroids[0].len();
            let mut new_centroids: Vec<Vec<D>> = vec![vec![D::zero(); dims]; self.k];
            let mut counts = vec![0; self.k];

            for (idx, point) in self.data.iter().enumerate() {
                let label = labels[idx];
                for (dim, value) in point.iter().enumerate() {
                    new_centroids[label][dim] = new_centroids[label][dim] + *value;
                }
                counts[label] += 1;
            }

            for (centroid_idx, centroid) in new_centroids.iter_mut().enumerate() {
                if counts[centroid_idx] != 0 {
                    let count_as_d = D::from_usize(counts[centroid_idx]).unwrap();
                    for value in centroid.iter_mut() {
                        *value = *value / count_as_d;
                    }
                }
            }

            centroids = new_centroids;
        }
        self.clusters = labels.clone();
        labels
    }

    /// Maps labels to the generated clusters in fit call
    /// 
    /// # Arguments 
    /// 
    /// * `labels`: A 1d array of labels representing the data provided to fit
    /// 
    /// returns: HashMap<usize, L, RandomState> 
    /// 
    /// # Examples 
    /// 
    /// ```
    /// use rusty_science::clustering::KMeansCluster;
    /// let mut knn = KMeansCluster::<f64>::new(3);
    /// let data = vec![vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 4.0]];
    /// let labels = vec![1, 1, 2];
    /// let clusters = knn.fit(data);
    /// let labeled_clusters = knn.map_cluster_to_label(labels);
    /// ```
    pub fn map_cluster_to_label<L>(&self, labels: Vec<L>) -> HashMap<usize, L>
    where
        L: Eq + Hash + Clone,
    {
        let mut label_map: HashMap<usize, L> = HashMap::new();
        let mut cluster_label_counts: HashMap<usize, HashMap<L, usize>> = HashMap::new();

        // Iterate over each data point and update label counts per cluster
        for (idx, &cluster_id) in self.clusters.iter().enumerate() {
            let label = &labels[idx];
            let label_counts = cluster_label_counts
                .entry(cluster_id)
                .or_insert_with(HashMap::new);
            *label_counts.entry(label.clone()).or_insert(0) += 1;
        }

        // Determine the most frequent label for each cluster
        for (cluster_id, counts) in cluster_label_counts.iter() {
            if let Some((most_common_label, _)) = counts.iter().max_by_key(|&(_, count)| count) {
                label_map.insert(*cluster_id, most_common_label.clone());
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
        let mut clusterer = KMeansCluster::<f64>::new(2);
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
        let mut clusterer = KMeansCluster::<f64>::new(1);
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
        let distance = _calculate_distance(&data[0], &data[1], &*"euclidean".to_string());
        assert_eq!(distance, 5.0);
    }

    #[test]
    fn test_map_cluster_to_label() {
        let mut knn_cluster = KMeansCluster::<f64>::new(2);

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
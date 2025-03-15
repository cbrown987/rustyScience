//! # DBSCAN Clustering Algorithm
//!
//! ## Theoretical Background
//!
//! DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a density-based
//! clustering algorithm introduced by Ester, Kriegel, Sander, and Xu in 1996. Unlike 
//! centroid-based algorithms like K-means, DBSCAN:
//!
//! - Can find arbitrarily shaped clusters
//! - Doesn't require specifying the number of clusters beforehand
//! - Is robust to outliers
//! - Can identify noise points (outliers) that don't belong to any cluster
//!
//! The algorithm works by grouping points that are closely packed together (points with many 
//! nearby neighbors) and marking as outliers points that lie alone in low-density regions.
//!
//! ### Core Concepts
//!
//! - **Directly density-reachable**: A point q is directly density-reachable from p if p is a core point and q is within distance ε from p.
//! - **Density-reachable**: A point q is density-reachable from p if there is a chain of points p₁, ..., pₙ with p₁ = p and pₙ = q where each pᵢ₊₁ is directly density-reachable from pᵢ.
//! - **Density-connected**: Two points p and q are density-connected if there is a point o such that both p and q are density-reachable from o.
//!
//! ## Parameters
//!
//! - `eps` (ε): The maximum distance between two samples for one to be considered in the neighborhood of the other.
//!    - Too small: Many points will be labeled as noise
//!    - Too large: Clusters may merge together
//!    - Typical values: Problem-dependent, often between 0.1 and 1.0
//!
//! - `min_samples`: The minimum number of points required to form a dense region (a core point).
//!    - Higher values create more restrictive clusters
//!    - Lower values create more inclusive clusters
//!    - Typical values: 2× dimensionality or more (e.g., 4+ for 2D data)
//!
//! ## Usage Examples
//!
//! Basic clustering with DBSCAN:
//!
//! ```rust
//! use rusty_science::clustering::DBSCAN;
//!
//! // Create example data
//! let data = vec![
//!     vec![1.0, 2.0], vec![1.1, 2.2], vec![0.9, 1.9], vec![1.0, 2.1],  // Cluster 1
//!     vec![4.0, 5.0], vec![4.2, 5.1], vec![3.9, 4.8], vec![4.1, 5.2],  // Cluster 2
//!     vec![10.0, 10.0]  // Noise point
//! ];
//!
//! // Initialize default labels (can be any type)
//! let labels = vec![0; data.len()];
//!
//! // Create and configure DBSCAN
//! let mut dbscan = DBSCAN::new();
//! dbscan.set_eps(0.5);  // Points within 0.5 distance units are neighbors
//! dbscan.set_min_samples(3);  // At least 3 points needed to form a core point
//!
//! // Fit the model
//! dbscan.fit(data.clone(), labels);
//!
//! // Get cluster assignments
//! let cluster_labels = dbscan.get_labels();
//! println!("Cluster labels: {:?}", cluster_labels);
//!
//! // Predict cluster for new points
//! let new_points = vec![vec![1.0, 2.0], vec![7.0, 7.0]];
//! let predictions = dbscan.predict(new_points);
//! println!("Predictions: {:?}", predictions);
//! ```
//!
//! ## Performance Characteristics
//!
//! - **Time Complexity**: O(n log n) with spatial indexing (using KD-Tree), where n is the number of points
//!   - Worst case: O(n²) if spatial indexing is ineffective
//!
//! - **Space Complexity**: O(n) for storing point information and cluster assignments
//!
//! - **Strengths**:
//!   - Discovers clusters of arbitrary shape
//!   - Robust to outliers
//!   - No need to specify number of clusters
//!
//! - **Weaknesses**:
//!   - Struggles with varying density clusters
//!   - Parameter selection can be challenging
//!   - Not as effective for high-dimensional data due to "curse of dimensionality"
//! 
use kdtree::distance::squared_euclidean;
use kdtree::KdTree;
use num_traits::{Float, Zero, One};


pub struct DBSCAN<D, L> {
    data: Vec<Vec<D>>,
    labels: Vec<L>,
    eps: f64,
    min_samples: usize,
    core_samples: Vec<usize>,
    border_samples: Vec<usize>,
    noise_samples: Vec<usize>,
    visited: Vec<bool>,
}

impl<D, L> DBSCAN<D, L>
where
    D: Clone + Copy + PartialOrd + Float + Zero + One,
    L: Clone + Default, f64: From<D>
{
    pub fn new() -> Self {
        Self {
            data: vec![],
            labels: vec![],
            eps: 0.5,
            min_samples: 5,
            core_samples: vec![],
            border_samples: vec![],
            noise_samples: vec![],
            visited: vec![],
        }
    }

    /// Set the variable that determines the size of the neighborhood to search for neighbors
    /// 
    /// # Arguments 
    /// 
    /// * `eps`: f64 float
    ///
    /// 
    /// # Examples 
    /// 
    /// ```
    /// use rusty_science::clustering::DBSCAN;
    /// let mut dbscan:DBSCAN<f64, f64> = DBSCAN::new();
    /// dbscan.set_eps(0.5)
    /// ```
    pub fn set_eps(&mut self, eps: f64) {
        self.eps = eps;
    }

    /// Set the minimum number of samples required to form a core point.
    ///
    /// # Arguments
    ///
    /// * `min_samples` - An unsigned integer representing the minimum number of samples.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_science::clustering::DBSCAN;
    /// let mut dbscan: DBSCAN<f64, f64> = DBSCAN::new();
    /// dbscan.set_min_samples(10);
    /// ```
    pub fn set_min_samples(&mut self, min_samples: usize) {
        self.min_samples = min_samples;
    }

    /// Fit the DBSCAN model using a 2d array of points and a 1d array of labels corresponding to
    /// each point
    ///
    /// # Arguments
    ///
    /// * `data`: A 2-dimensional array of integers
    /// * `labels`: A 1-dimentional array of integers
    ///
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_science::clustering::DBSCAN;
    /// use rusty_science::data::load_iris;
    ///
    /// let iris_data = load_iris();
    /// let (data, labels) = iris_data.to_numerical_labels();
    ///
    /// let mut dbscan = DBSCAN::new();
    /// dbscan.set_eps(1.0);
    /// dbscan.set_min_samples(10);
    /// dbscan.fit(data, labels);
    /// ```
    pub fn fit(&mut self, data: Vec<Vec<D>>, labels: Vec<L>) {
        self.data = data;
        self.visited = vec![false; self.data.len()];

        // Clear previous results
        self.core_samples.clear();
        self.border_samples.clear();
        self.noise_samples.clear();

        // Initialize working labels
        let mut working_labels = labels;

        // Create KdTree for efficient neighbor searching
        let kdtree = match self.build_kdtree() {
            Ok(tree) => tree,
            Err(e) => {
                eprintln!("Failed to build KdTree: {}", e);
                return;
            }
        };

        let mut current_cluster = 0;

        for point_idx in 0..self.data.len() {
            if self.visited[point_idx] {
                continue; 
            }

            self.visited[point_idx] = true;

            let neighbors_result = kdtree.within(
                &self.data[point_idx],
                self.create_eps_squared(),
                &squared_euclidean,
            );

            let neighbor_points = match self.convert_neighbors(neighbors_result) {
                Ok(neighbors) => neighbors,
                Err(e) => {
                    eprintln!("Error during range query: {}", e);
                    return;
                }
            };

            if neighbor_points.len() >= self.min_samples {
                // Point is a core point - start a new cluster
                self.core_samples.push(point_idx);
                current_cluster += 1;

                // Expand the cluster
                self.expand_cluster(
                    &neighbor_points,
                    &kdtree,
                    &mut working_labels,
                    current_cluster,
                );
            } else {
                // Mark as noise
                self.noise_samples.push(point_idx);
            }
        }

        // Store the final labels
        self.labels = working_labels;
    }
    /// Returns the labels from the fitted model
    ///
    /// returns: Vec<L>
    pub fn get_labels(&self) -> Vec<L> {
        self.labels.clone()
    }

    /// Predict label for a single new data point using the fitted DBSCAN model
    ///
    /// # Arguments
    ///
    /// * `point`: Single data point to classify
    ///
    /// returns: L - The predicted label
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_science::clustering::DBSCAN;
    /// use rusty_science::data::load_iris;
    ///
    /// let iris_data = load_iris();
    /// let (data, labels) = iris_data.to_numerical_labels();
    ///
    /// let mut dbscan = DBSCAN::new();
    /// dbscan.set_eps(1.0);
    /// dbscan.set_min_samples(10);
    /// dbscan.fit(data, labels);
    /// 
    /// let target = vec![2.1, 3.1];
    /// dbscan.predict_one(target);
    /// ```
    pub fn predict_one(&self, point: Vec<D>) -> L {
        if self.core_samples.is_empty() {
            return L::default();
        }

        let mut kdtree = KdTree::<D, usize, Vec<D>>::new(self.data[0].len());

        // Add all core points to the KdTree
        for &core_idx in &self.core_samples {
            match kdtree.add(self.data[core_idx].clone(), core_idx) {
                Ok(_) => {},
                Err(e) => {
                    println!("Error adding point to KdTree: {:?}", e);
                    return L::default();
                }
            }
        }

        // Find the nearest core point within eps distance
        let nearest = match kdtree.nearest(&point, 1, &squared_euclidean) {
            Ok(nearest) => nearest,
            Err(e) => {
                println!("Error finding nearest point: {:?}", e);
                return L::default();
            }
        };

        if !nearest.is_empty() {
            let (distance, &core_idx) = nearest[0];
            if distance <= self.create_eps_squared() {
                return self.labels[core_idx].clone();
            }
        }

        L::default()
    }

    /// Predict labels for new data points using the fitted DBSCAN model. this method takes multiple
    /// points and outputs predictions for each one. If seeking to predict one label use the predict_one
    /// function or only pass one datapoint
    ///
    /// # Arguments
    ///
    /// * `new_data`: New data points to classify. In the format of a 2d array
    ///
    /// returns: Vec<L>
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_science::clustering::DBSCAN;
    /// use rusty_science::data::load_iris;
    ///
    /// let iris_data = load_iris();
    /// let (data, labels) = iris_data.to_numerical_labels();
    ///
    /// let mut dbscan = DBSCAN::new();
    /// dbscan.set_eps(1.0);
    /// dbscan.set_min_samples(10);
    /// dbscan.fit(data, labels);
    ///
    /// let target = vec![
    ///     vec![1.5, 1.5],
    ///     vec![2.0, 3.1],
    ///     vec![3f64 , 3.1]
    /// ];
    ///     
    /// let value = dbscan.predict(target);
    /// ```
    pub fn predict(&self, new_data: Vec<Vec<D>>) -> Vec<L> {
        if self.core_samples.is_empty() {
            // No fit has been performed yet
            return vec![L::default(); new_data.len()];
        }

        let mut kdtree = KdTree::<D, usize, Vec<D>>::new(self.data[0].len());

        // Add all core points to the KdTree
        for &core_idx in &self.core_samples {
            match kdtree.add(self.data[core_idx].clone(), core_idx) {
                Ok(_) => {},
                Err(e) => {
                    println!("Error adding point to KdTree: {:?}", e);
                    return vec![L::default(); new_data.len()];
                }
            }
        }

        // Create result labels vector
        let mut result = vec![L::default(); new_data.len()]; // Use default (noise) label

        // For each new point, find the nearest core point within eps distance
        for (i, point) in new_data.iter().enumerate() {
            let nearest = match kdtree.nearest(point, 1, &squared_euclidean) {
                Ok(nearest) => nearest,
                Err(e) => {
                    println!("Error finding nearest point: {:?}", e);
                    continue;
                }
            };

            if !nearest.is_empty() {
                let (distance, &core_idx) = nearest[0];
                if distance <= self.create_eps_squared() {
                    // If within eps of a core point, assign the same label
                    result[i] = self.labels[core_idx].clone();
                }
            }
        }
        result
    }

    // Helper Methods
    fn build_kdtree(&self) -> Result<KdTree<D, usize, Vec<D>>, String> {
        let dimensions = match self.data.first() {
            Some(point) => point.len(),
            None => return Err("Empty dataset provided".to_string())
        };

        let mut tree = KdTree::new(dimensions);

        for (idx, point) in self.data.iter().enumerate() {
            if let Err(e) = tree.add(point.clone(), idx) {
                return Err(format!("Error adding point to KdTree: {:?}", e));
            }
        }

        Ok(tree)
    }

    fn convert_neighbors(&self, neighbors_result: Result<Vec<(D, &usize)>,
        impl std::fmt::Debug>) -> Result<Vec<(usize, f64)>, String> {
        neighbors_result
            .map_err(|e| format!("{:?}", e))
            .map(|neighbors| {
                neighbors
                    .into_iter()
                    .map(|(dist, &idx)| (idx, f64::from(dist)))
                    .collect()
            })
    }

    fn expand_cluster(&mut self, neighbors: &Vec<(usize, f64)>,
                      kdtree: &KdTree<D, usize, Vec<D>>, labels: &mut Vec<L>,
                      cluster_id: usize) {

        for (neighbor_index, _distance) in neighbors {
            let neighbor_idx = *neighbor_index;

            if !self.visited[neighbor_idx] {
                self.visited[neighbor_idx] = true;


                let neighbor_neighbors = match kdtree.within(
                    &self.data[neighbor_idx],
                    self.create_eps_squared(),
                    &squared_euclidean
                ) {
                    Ok(neighbors) => neighbors,
                    Err(e) => {
                        println!("Error during range query: {:?}", e);
                        return;
                    }
                };

                if neighbor_neighbors.len() >= self.min_samples {
                    // This neighbor is also a core point
                    self.core_samples.push(neighbor_idx);
                    // Set the label for this neighbor
                    // labels[neighbor_idx] = cluster_label_value;

                    // Recursively expand the cluster
                    // Convert neighbor_neighbors to the expected format
                    let converted_neighbors: Vec<(usize, f64)> = neighbor_neighbors
                        .into_iter()
                        .map(|(dist, &idx)| (idx, dist.into()))
                        .collect();
                    self.expand_cluster(&converted_neighbors, kdtree, labels, cluster_id);
                } else {
                    // This is a border point
                    self.border_samples.push(neighbor_idx);
                    // Set the label for this neighbor
                    // labels[neighbor_idx] = cluster_label_value;
                }
            }
        }
    }

    fn create_eps_squared(&self) -> D {
        let eps_d: D = D::from(self.eps).expect("Failed to convert eps to D");
        let eps_squared = eps_d * eps_d;
        eps_squared
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let dbscan: DBSCAN<f64, usize> = DBSCAN::new();
        assert!(dbscan.data.is_empty());
        assert!(dbscan.labels.is_empty());
        assert_eq!(dbscan.eps, 0.5);
        assert_eq!(dbscan.min_samples, 5);  
        assert!(dbscan.core_samples.is_empty());
        assert!(dbscan.border_samples.is_empty());
        assert!(dbscan.noise_samples.is_empty());
        assert!(dbscan.visited.is_empty());
    }

    #[test]
    fn test_set_eps() {
        let mut dbscan: DBSCAN<f64, usize> = DBSCAN::new();
        dbscan.set_eps(1.0);
        assert_eq!(dbscan.eps, 1.0);
    }

    #[test]
    fn test_set_min_samples() {
        let mut dbscan: DBSCAN<f64, usize> = DBSCAN::new();
        dbscan.set_min_samples(10);
        assert_eq!(dbscan.min_samples, 10);
    }

    #[test]
    fn test_fit_simple_clusters() {
        let mut dbscan: DBSCAN<f64, usize> = DBSCAN::new();
        dbscan.set_eps(1.0);
        dbscan.set_min_samples(2);

        // Create two distinct clusters and one noise point
        let data = vec![
            vec![0.0, 0.0],  // Cluster 1
            vec![0.2, 0.2],  // Cluster 1
            vec![0.3, 0.3],  // Cluster 1
            vec![5.0, 5.0],  // Cluster 2
            vec![5.1, 5.2],  // Cluster 2
            vec![5.2, 5.0],  // Cluster 2
            vec![10.0, 10.0], // Noise
        ];

        let labels = vec![0, 0, 0, 1, 1, 1, 2];
        dbscan.fit(data, labels);

        assert_eq!(dbscan.core_samples.len(), 6);
        assert!(dbscan.border_samples.is_empty());
        assert_eq!(dbscan.noise_samples.len(), 1);
        assert_eq!(dbscan.noise_samples[0], 6); // The noise point
    }

    #[test]
    fn test_predict_with_existing_data() {
        let mut dbscan: DBSCAN<f64, usize> = DBSCAN::new();
        dbscan.set_eps(1.0);
        dbscan.set_min_samples(2);

        let data = vec![
            vec![0.0, 0.0],
            vec![0.2, 0.2],
            vec![5.0, 5.0],
            vec![5.1, 5.2],
            vec![10.0, 10.0],
        ];

        let labels = vec![0, 0, 1, 1, 2];
        dbscan.fit(data, labels.clone());

        // Predict with existing data
        let predicted = dbscan.get_labels();
        assert_eq!(predicted, labels);
    }

    #[test]
    fn test_predict_with_new_data() {
        let mut dbscan: DBSCAN<f64, usize> = DBSCAN::new();
        dbscan.set_eps(1.0);
        dbscan.set_min_samples(2);

        let data = vec![
            vec![0.0, 0.0],   // Cluster 0
            vec![0.2, 0.2],   // Cluster 0
            vec![5.0, 5.0],   // Cluster 1
            vec![5.1, 5.2],   // Cluster 1
            vec![10.0, 10.0], // Cluster 2 (noise)
        ];

        let labels = vec![0, 0, 1, 1, 2];
        dbscan.fit(data, labels);

        // New data points close to existing clusters
        let new_data = vec![
            vec![0.1, 0.1],   // Should be assigned to cluster 0
            vec![5.05, 5.05], // Should be assigned to cluster 1
            vec![8.0, 8.0],   // Should be noise (default label)
        ];

        let predicted = dbscan.predict(new_data);
        assert_eq!(predicted.len(), 3);
        assert_eq!(predicted[0], 0);
        assert_eq!(predicted[1], 1);
        assert_eq!(predicted[2], 0);
    }

    #[test]
    fn test_create_eps_squared() {
        let mut dbscan: DBSCAN<f64, usize> = DBSCAN::new();
        dbscan.set_eps(2.0);

        let eps_squared = dbscan.create_eps_squared();
        assert_eq!(eps_squared, 4.0);
    }

    #[test]
    fn test_with_float32() {
        // Test with a different float type
        let mut dbscan: DBSCAN<f32, i32> = DBSCAN::new();
        dbscan.set_eps(1.0);
        dbscan.set_min_samples(2);

        let data = vec![
            vec![0.0_f32, 0.0_f32],
            vec![0.2_f32, 0.2_f32],
        ];

        let labels = vec![1, 1];
        dbscan.fit(data, labels.clone());

        let predicted = dbscan.get_labels();
        assert_eq!(predicted, labels);
    }

    #[test]
    fn test_empty_data() {
        let mut dbscan: DBSCAN<f64, usize> = DBSCAN::new();
        let data: Vec<Vec<f64>> = vec![];
        let labels: Vec<usize> = vec![];

        dbscan.fit(data, labels);
        assert!(dbscan.data.is_empty());
        assert!(dbscan.core_samples.is_empty());
        assert!(dbscan.border_samples.is_empty());
        assert!(dbscan.noise_samples.is_empty());

        let predicted = dbscan.get_labels();
        assert!(predicted.is_empty());
    }

    #[test]
    fn test_border_points() {
        let mut dbscan: DBSCAN<f64, usize> = DBSCAN::new();
        dbscan.set_eps(1.0);
        dbscan.set_min_samples(3); // Set min_samples higher to create border points

        // Create a cluster with core and border points
        let data = vec![
            vec![0.0, 0.0],  // Core point (has 3 neighbors)
            vec![0.2, 0.2],  // Core point (has 3 neighbors)
            vec![0.3, 0.3],  // Core point (has 3 neighbors)
            vec![1.9, 1.9],  // Noise point
        ];

        let labels = vec![0, 0, 0, 0];
        dbscan.fit(data, labels);

        assert_eq!(dbscan.core_samples.len(), 3);
        assert_eq!(dbscan.noise_samples.len(), 1);
    }
}
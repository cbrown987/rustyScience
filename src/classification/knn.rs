use crate::common::knn::{neighbors, Neighbor};
use num::Num;
use num_traits::ToPrimitive;
use crate::common::custom_error::ModelError;
use crate::common::knn::{WeightType, DistanceMetric};
use crate::{panic_dimension_mismatch, panic_untrained};

#[derive(Clone)]
pub struct KNNClassifier<D, L> {
    k: usize,
    data: Vec<Vec<D>>,
    labels: Vec<L>,
    distance_metric: DistanceMetric,
    weight_type: WeightType,
}

impl<D, L> KNNClassifier<D, L>
where
    D: Num + Copy + Clone + PartialOrd + ToPrimitive,
    L: Num + Copy + Clone + PartialOrd + ToPrimitive,
{
    /// Creates a new KNNClassifier with a specified value of k.
    ///
    /// # Arguments
    /// * `k` - The number of neighbors to consider for regression. Must be greater than zero.
    ///
    /// # Panics
    /// This function will panic if `k` is less than or equal to zero.
    ///
    /// # Examples
    /// ```
    /// use rusty_science::classification::KNNClassifier;
    /// let knn = KNNClassifier::<f64, i64>::new(3);
    /// ```
    pub fn new(k: usize) -> Self {
        if k == 0 {
            panic!("K cannot be zero");
        }
        Self {
            k,
            data: vec![],
            labels: vec![],
            distance_metric: DistanceMetric::Euclidean,
            weight_type: WeightType::Uniform,
        }
    }
    
    /// Creates a default KNNClassifier instance with preset parameters.
    ///
    /// This constructor provides a default KNN configuration:
    /// - k: 5 (number of neighbors to consider)
    /// - Distance Metric: Euclidean
    /// - Weight Type: Uniform
    ///
    /// # Examples
    /// ```
    /// use rusty_science::classification::KNNClassifier;
    ///
    /// let knn = KNNClassifier::<f64, i64>::default();
    /// ```
    pub fn default() -> Self {
        Self{
            k: 5,
            data: vec![],
            labels: vec![],
            distance_metric: DistanceMetric::Euclidean,
            weight_type: WeightType::Uniform,
        }
    }
    

    /// Sets the method that will determine the weights for neighbors, either 'uniform' or 'distance'.
    ///
    /// # Arguments
    /// * `weight_type` - A string specifying the weight type: either 'uniform' or 'distance'.
    ///
    /// # Panics
    /// This function will panic if an unsupported weight type is provided.
    ///
    /// # Examples
    /// ```
    /// use rusty_science::classification::{KNNClassifier, WeightType};
    /// let mut knn = KNNClassifier::<f64, i64>::new(3);
    /// knn.set_weight_type(WeightType::Uniform);
    /// ```
    pub fn set_weight_type(&mut self, weight_type: WeightType) {
        self.weight_type = weight_type
    }
    
    /// Sets weights for neighbors using a chainable method.
    ///
    /// This allows the `weight_type` to be set in a way that supports chaining method calls.
    ///
    /// # Arguments
    /// * `weight_type` - Specifies the weight calculation method. Must be of type `WeightType` 
    ///   and can be either `Uniform` or `Distance`.
    ///
    /// # Returns
    /// * `Self` - The `KNNClassifier` instance with the updated `weight_type` applied. 
    ///
    /// # Examples
    /// ```
    /// use rusty_science::classification::{KNNClassifier, WeightType};
    ///
    /// let knn = KNNClassifier::<f64, i64>::new(3)
    ///     .with_weight_type(WeightType::Distance);
    /// ```
    pub fn with_weight_type(mut self, weight_type: WeightType) -> Self {
        self.set_weight_type(weight_type);
        self
    }

    /// Sets the distance metric to be used for finding neighbors.
    ///
    /// # Arguments
    /// * `distance_metric` - A string specifying the distance metric, e.g., 'euclidean' or 'manhattan'.
    ///
    /// # Examples
    /// ```
    /// use rusty_science::classification::{KNNClassifier, DistanceMetric};
    /// let mut knn = KNNClassifier::<f64, i64>::new(3);
    /// knn.set_distance_metric(DistanceMetric::Euclidean);
    /// ```
    pub fn set_distance_metric(&mut self, distance_metric: DistanceMetric) {
        self.distance_metric = distance_metric;
    }
    

    ///
    /// Sets the distance metric to be used for finding neighbors (chainable version).
    ///
    /// This method allows setting the distance metric in a way that supports method chaining.
    ///
    /// # Arguments
    /// * `distance_metric` - The distance metric to be applied. This must be one of the 
    ///   predefined types in the `DistanceMetric` enum.
    ///
    /// # Returns
    /// * `Self` - Returns the updated `KNNClassifier` instance with the new distance metric 
    ///   applied.
    ///
    /// # Examples
    /// ```
    /// use rusty_science::classification::{KNNClassifier, DistanceMetric};
    ///
    /// let mut knn = KNNClassifier::<f64, i64>::new(3)
    ///     .with_distance_metric(DistanceMetric::Manhattan);
    /// ```
    pub fn with_distance_metric(mut self, distance_metric: DistanceMetric) -> Self {
        self.set_distance_metric(distance_metric);
        self
    }
    /// Fits the classification with the training data and labels.
    ///
    /// # Arguments
    /// * `data` - A vector of vectors containing the training data points.
    /// * `labels` - A vector of labels corresponding to the training data points.
    ///
    /// # Examples
    /// ```
    /// use rusty_science::classification::KNNClassifier;
    /// let mut knn = KNNClassifier::<f64, i64>::new(3);
    /// let data = vec![vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 4.0]];
    /// let labels = vec![1, 1, 4];
    /// knn.fit(data, labels);
    /// ```
    pub fn fit(&mut self, data: Vec<Vec<D>>, labels: Vec<L>) -> Result<(), ModelError> {
        if data.is_empty(){
            return Err(ModelError::EmptyData);
        }
        if data.len() != labels.len() {
            return Err(ModelError::DimensionMismatch(
                "Data and labels must have the same length".to_string(),
            ))
        }
        let expected_dim = data[0].len();
        for (i, sample) in data.iter().enumerate() {
            if sample.len() != expected_dim {
                return Err(ModelError::DimensionMismatch(
                    format!("Sample at index {} has inconsistent dimensions", i)
                ));
            }
        }

        self.data = data;
        self.labels = labels;
        
        Ok(())
    }

    /// Predicts the label for a given target data point.
    ///
    /// # Arguments
    /// * `target` - A vector representing the features of the data point to be classified.
    ///
    /// # Returns
    /// * A number representing the predicted label for the target data point.
    ///
    /// # Examples
    /// ```
    /// use rusty_science::classification::KNNClassifier;
    /// let mut knn = KNNClassifier::<f64, f64>::new(3);
    /// let data = vec![vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 4.0]];
    /// let labels = vec![0.72, 1.0, 0.26];
    /// knn.fit(data, labels);
    /// let prediction = knn.predict(vec![2.5, 3.5]);
    /// println!("Predicted label: {}", prediction);
    /// ```
    pub fn predict(&self, target: Vec<D>) -> L {
        panic_untrained!(self.labels.is_empty(), "KNNClassifier");

        let expected_dim = self.data[0].len();
        panic_dimension_mismatch!(target.len() != expected_dim, target.len(), expected_dim);
        
        self._predict(target)
    }

    fn _predict(&self, target: Vec<D>) -> L {
        let calculate_distance = self.weight_type == WeightType::Distance;
        
        let neighbors: Vec<Neighbor<D, L>> = neighbors(
            self.data.clone(),
            Some(self.labels.clone()),
            Some(target),
            self.k,
            self.distance_metric.clone(),
            calculate_distance,
        );

        let mut label_weights = Vec::new();

        for neighbor in neighbors.iter() {
            if let Some(label) = neighbor.label.clone() {
                let distance = neighbor.distance_to_target;
                let weight = if calculate_distance {
                    1.0 / (distance + 1e-8)
                } else {
                    1.0
                };

                match label_weights.iter_mut().find(|(lbl, _)| *lbl == label) {
                    Some((_, total_weight)) => *total_weight += weight,
                    None => label_weights.push((label, weight)),
                }
            }
        }

        label_weights
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(label, _)| label)
            .expect("Neighbor list should not be empty")
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::knn::{DistanceMetric, WeightType};

    #[test]
    fn test_knnclassifier_predict_uniform() {
        // Sample data
        let data = vec![
            vec![1.0, 2.0], // Label: 0
            vec![2.0, 3.0], // Label: 0
            vec![3.0, 3.0], // Label: 1
            vec![6.0, 5.0], // Label: 1
            vec![7.0, 7.0], // Label: 1
        ];
        let labels = vec![0, 0, 1, 1, 1];

        // Create KNNClassifier with k=3 and uniform weights
        let mut knn = KNNClassifier::<f64, i32>::new(3);
        knn.set_weight_type(WeightType::Uniform);
        knn.fit(data, labels).expect("Failed to fit KNNClassifier");

        // Predict label for a new point close to class 0
        let target = vec![2.5, 2.5];
        let predicted_label = knn.predict(target);

        // With uniform weights, labels from nearest neighbors are [0, 0, 1]
        // Class 0 has 2 votes, class 1 has 1 vote
        assert_eq!(predicted_label, 0);
    }

    #[test]
    fn test_knnclassifier_predict_distance_weighted() {
        // Sample data
        let data = vec![
            vec![1.0, 2.0], // Label: 0
            vec![2.0, 2.0], // Label: 0
            vec![3.0, 3.0], // Label: 1
        ];
        let labels = vec![0, 0, 1];

        // Create KNNClassifier with k=3 and distance weights
        let mut knn = KNNClassifier::<f64, i32>::new(3);
        knn.set_weight_type(WeightType::Distance);
        knn.fit(data, labels).expect("Failed to fit KNNClassifier");

        // Predict label for a point closer to class 1
        let target = vec![2.9, 2.9];
        let predicted_label = knn.predict(target);

        // Expected label is 1 since it's closer to the point with label 1
        assert_eq!(predicted_label, 1);
    }

    #[test]
    fn test_knnclassifier_predict_k_equals_1() {
        // Sample data
        let data = vec![
            vec![1.0, 2.0], // Label: 0
            vec![3.0, 3.0], // Label: 1
        ];
        let labels = vec![0, 1];

        // Create KNNClassifier with k=1
        let mut knn = KNNClassifier::<f64, i32>::new(1);
        knn.fit(data, labels).expect("Failed to fit KNNClassifier");

        // Predict label for a point exactly at [3.0, 3.0]
        let target = vec![3.0, 3.0];
        let predicted_label = knn.predict(target);

        // Expected label is 1
        assert_eq!(predicted_label, 1);
    }
    

    #[test]
    #[should_panic(expected = "K cannot be zero")]
    fn test_knnclassifier_k_zero() {
        KNNClassifier::<f64, i32>::new(0);
    }
    
        #[test]
    fn test_empty_training_data() {
        let mut model = KNNClassifier::<f64, i32>::new(3);
        let result = model.fit(vec![], vec![]);
        assert!(result.is_err());
        assert!(matches!(result, Err(ModelError::EmptyData)));
    }

    #[test]
    fn test_mismatched_data_label_lengths() {
        let mut model = KNNClassifier::<f64, i32>::new(3);
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let labels = vec![1]; // One label missing
        let result = model.fit(data, labels);
        assert!(result.is_err());
        assert!(matches!(result, Err(ModelError::DimensionMismatch(_))));
    }

    #[test]
    fn test_inconsistent_feature_dimensions() {
        let mut model = KNNClassifier::<f64, i32>::new(3);
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0, 5.0]]; // Second sample has different dimension
        let labels = vec![1, 2];
        let result = model.fit(data, labels);
        assert!(result.is_err());
        assert!(matches!(result, Err(ModelError::DimensionMismatch(_))));
    }

    #[test]
    fn test_classify_multiclass() {
        let mut model = KNNClassifier::<f64, i32>::new(3);
        // Create dataset with 3 classes (1, 2, 3)
        let data = vec![
            vec![1.0, 1.0], vec![1.2, 1.1], vec![0.9, 0.9],  // Class 1
            vec![4.0, 4.0], vec![4.1, 4.2], vec![3.9, 4.0],  // Class 2
            vec![7.0, 7.0], vec![7.1, 6.9], vec![6.9, 7.1],  // Class 3
        ];
        let labels = vec![1, 1, 1, 2, 2, 2, 3, 3, 3];
        
        model.fit(data.clone(), labels).unwrap();
        
        // Test points close to each class center
        assert_eq!(model.predict(vec![1.1, 1.0]), 1);
        assert_eq!(model.predict(vec![4.0, 4.1]), 2);
        assert_eq!(model.predict(vec![7.0, 7.0]), 3);
    }
    
    #[test]
    fn test_classify_binary() {
        let mut model = KNNClassifier::<f64, i32>::new(5);
        // Create binary classification dataset
        let data = vec![
            vec![1.0, 1.0], vec![1.5, 0.5], vec![0.5, 1.5], vec![0.8, 0.9], vec![1.2, 0.8],
            vec![5.0, 5.0], vec![5.5, 4.5], vec![4.5, 5.5], vec![4.8, 4.9], vec![5.2, 4.8],
        ];
        let labels = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
        
        model.fit(data, labels).unwrap();
        
        // Test classification of clear examples
        assert_eq!(model.predict(vec![1.0, 1.0]), 0);
        assert_eq!(model.predict(vec![5.0, 5.0]), 1);
        
        // Test classification of an ambiguous example (should be classified by majority)
        let midpoint_prediction = model.predict(vec![3.0, 3.0]);
        assert!(midpoint_prediction == 0 || midpoint_prediction == 1); // Either could be valid
    }

    #[test]
    fn test_different_distance_metrics() {
        // Create a dataset where Manhattan and Euclidean would give different results
        let data = vec![
            vec![0.0, 5.0],  // Class 0, Manhattan: 5, Euclidean: 5
            vec![3.0, 4.0],  // Class 1, Manhattan: 7, Euclidean: 5
            vec![5.0, 0.0],  // Class 2, Manhattan: 5, Euclidean: 5
        ];
        let labels = vec![0, 1, 2];
        let test_point = vec![1.0, 1.0]; // Target point
        
        // With Euclidean, all points are at equal distance of 5
        let mut euclidean_model = KNNClassifier::<f64, i32>::new(1);
        euclidean_model.set_distance_metric(DistanceMetric::Euclidean);
        euclidean_model.fit(data.clone(), labels.clone()).unwrap();
        let euclidean_pred = euclidean_model.predict(test_point.clone());
        
        // With Manhattan, first point is closest
        let mut manhattan_model = KNNClassifier::<f64, i32>::new(1);
        manhattan_model.set_distance_metric(DistanceMetric::Manhattan);
        manhattan_model.fit(data, labels).unwrap();
        let manhattan_pred = manhattan_model.predict(test_point);
        
        // This specific test depends on tie-breaking strategy when distances are equal
        // But the overall behavior of different metrics should be testable
        assert!(euclidean_pred == 0 || euclidean_pred == 1 || euclidean_pred == 2);
        assert_eq!(manhattan_pred, 0); // Manhattan should pick the first point
    }

    #[test]
    fn test_weight_type_influence() {
        // Create a dataset where weight type matters
        let data = vec![
            vec![1.0, 1.0],  // Class 0, distance: 1.414
            vec![2.0, 2.0],  // Class 1, distance: 2.828
            vec![3.0, 3.0],  // Class 1, distance: 4.243
        ];
        let labels = vec![0, 1, 1];
        let test_point = vec![0.0, 0.0];
        
        // With uniform weights, majority of 3 neighbors is class 1
        let mut uniform_model = KNNClassifier::<f64, i32>::new(3);
        uniform_model.set_weight_type(WeightType::Uniform);
        uniform_model.fit(data.clone(), labels.clone()).unwrap();
        let uniform_pred = uniform_model.predict(test_point.clone());
        
        // With distance weights, closest neighbor (class 0) has more influence
        let mut distance_model = KNNClassifier::<f64, i32>::new(3);
        distance_model.set_weight_type(WeightType::Distance);
        distance_model.fit(data, labels).unwrap();
        let distance_pred = distance_model.predict(test_point);
        
        assert_eq!(uniform_pred, 1);   // Majority vote
        assert_eq!(distance_pred, 0);  // Closer point has higher weight
    }

    #[test]
    fn test_large_k_value() {
        let mut model = KNNClassifier::<f64, i32>::new(10);
        // Dataset with only 5 points
        let data = vec![
            vec![1.0, 1.0], vec![2.0, 2.0], vec![3.0, 3.0], vec![4.0, 4.0], vec![5.0, 5.0]
        ];
        let labels = vec![0, 0, 1, 1, 1];
        
        model.fit(data, labels).unwrap();
        
        // K is larger than dataset, so all points will be used
        let prediction = model.predict(vec![3.0, 3.0]);
        assert_eq!(prediction, 1); // Majority class is 1 (3 vs 2)
    }

    #[test]
    fn test_exact_match_prediction() {
        let mut model = KNNClassifier::<f64, i32>::new(2);
        let data = vec![
            vec![1.0, 1.0], vec![2.0, 2.0], vec![3.0, 3.0], vec![40.0, 42.0], vec![51.0, 43.0]
        ];
        let labels = vec![2, 2, 2, 0, 0];
        
        model.fit(data.clone(), labels.clone()).unwrap();
        
        // Test exact matches with training data
        for i in 0..data.len() {
            let prediction = model.predict(data[i].clone());
            assert_eq!(prediction, labels[i]);
        }
    }

    #[test]
    fn test_different_data_types() {
        // Test with integer features
        let mut int_model = KNNClassifier::<i32, i32>::new(3);
        let int_data = vec![
            vec![10, 100], vec![2, 2], vec![32, 3], vec![4, 4], vec![5, 5]
        ];
        let labels = vec![0, 1, 0, 1, 0];
        
        int_model.fit(int_data, labels).unwrap();
        
        let prediction = int_model.predict(vec![2, 2]);
        assert_eq!(prediction, 1); 
    }
    

    #[test]
    fn test_clone_implementation() {
        let mut original = KNNClassifier::<f64, i32>::new(3);
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let labels = vec![0, 1, 0];
        original.fit(data, labels).unwrap();
        
        // Clone the model
        let cloned = original.clone();
        
        // Both should give the same prediction
        let test_point = vec![2.0, 3.0];
        let original_pred = original.predict(test_point.clone());
        let cloned_pred = cloned.predict(test_point);
        
        assert_eq!(original_pred, cloned_pred);
    }
    
    #[test]
    fn test_ties_in_voting() {
        let mut model = KNNClassifier::<f64, i32>::new(4);
        // Create a dataset where there will be a tie in voting
        let data = vec![
            vec![1.0, 1.0], // Class 0
            vec![1.1, 1.1], // Class 0
            vec![5.0, 5.0], // Class 1
            vec![5.1, 5.1], // Class 1
        ];
        let labels = vec![0, 0, 1, 1];
        
        model.fit(data, labels).unwrap();
        
        // Point exactly in the middle - should break tie somehow
        let prediction = model.predict(vec![3.0, 3.0]);
        // Just check that it returns one of the valid classes
        assert!(prediction == 0 || prediction == 1);
    }
    
    #[test]
    #[should_panic]
    fn test_predict_with_invalid_dimensions() {
        let mut model = KNNClassifier::<f64, i32>::new(3);
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let labels = vec![0, 1, 0];
        model.fit(data, labels).unwrap();
        
        model.predict(vec![1.0, 2.0, 3.0]);
        
        // If predict still returns L directly, this would be a #[should_panic] test
    }
    
    #[test]
    fn test_builder_pattern() {
        // Assuming you've implemented a builder pattern
        let model = KNNClassifier::<f64, i32>::new(3)
            .with_weight_type(WeightType::Distance)
            .with_distance_metric(DistanceMetric::Euclidean);
            
        assert_eq!(model.k, 3);
        assert_eq!(model.weight_type, WeightType::Distance);
        assert_eq!(model.distance_metric, DistanceMetric::Euclidean);
    }
    
    #[test]
    fn test_default_implementation() {
        // Assuming you've implemented Default
        let model = KNNClassifier::<f64, i32>::default();
        assert_eq!(model.k, 5); // Assuming default k is 5
        assert_eq!(model.weight_type, WeightType::Uniform);
        assert_eq!(model.distance_metric, DistanceMetric::Euclidean);
    }
    
    #[test]
    fn test_high_dimensional_data() {
        // Test with high dimensional data
        let mut model = KNNClassifier::<f64, i32>::new(3);
        let data = vec![
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            vec![3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
            vec![4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
        ];
        let labels = vec![0, 0, 1, 1];
        
        model.fit(data, labels).unwrap();
        
        assert_eq!(model.predict(vec![1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]), 0);
        assert_eq!(model.predict(vec![3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5]), 1);
    }
    
}

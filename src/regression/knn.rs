use num_traits::{FromPrimitive, Num, ToPrimitive};
use crate::common::knn::{neighbors, DistanceMetric, Neighbor, WeightType};
use crate::common::custom_error::ModelError;
use crate::{panic_dimension_mismatch, panic_untrained};

#[derive(Clone)]
pub struct KNNRegression<D, L> {
    k: usize,
    data: Vec<Vec<D>>,
    labels: Vec<L>,
    weight_type: WeightType,
    distance_metric: DistanceMetric,
}

impl<D, L> KNNRegression<D, L>
where
    D: Num + Copy + Clone + PartialOrd + ToPrimitive,
    L: Num + Copy + Clone + PartialOrd + ToPrimitive + FromPrimitive,
{
    /// Creates a new KNNRegression with a specified value of k.
    ///
    /// # Arguments
    /// * `k` - The number of neighbors to consider for regression. Must be greater than zero.
    ///
    /// # Panics
    /// This function will panic if `k` is less than or equal to zero.
    ///
    /// # Examples
    /// ```
    /// use rusty_science::regression::KNNRegression;
    /// let knn = KNNRegression::<f64, f64>::new(3);
    /// ```
    pub fn new(k: usize) -> Self {
        if k <= 0 {
            panic!("K cannot be zero");
        }
        Self {
            k,
            data: vec![],
            labels: vec![],
            weight_type: WeightType::Uniform,
            distance_metric: DistanceMetric::Euclidean,
        }
    }
    
    /// Creates a default instance of `KNNRegression` with predefined parameters.
    ///
    /// The default values are:
    /// - `k`: 5 (number of neighbors),
    /// - `weight_type`: "uniform" (equal weights for all neighbors),
    /// - `distance_metric`: "euclidean" (Euclidean distance metric).
    ///
    /// # Example
    /// ```
    /// use rusty_science::regression::KNNRegression;
    /// let knn = KNNRegression::<f64, f64>::default();
    ///
    /// ```
    pub fn default() -> Self {
        Self{
            k: 5,
            data: vec![],
            labels: vec![],
            weight_type: WeightType::Uniform,
            distance_metric: DistanceMetric::Euclidean,
        }
    }



    /// Sets the method that will determine the weights for neighbors, either 'uniform' or 'distance'.
    /// Non builder method, doesnt return anything
    ///
    /// # Arguments
    /// * `weight_type` - A string specifying the weight type: either 'uniform' or 'distance'.
    ///
    /// # Panics
    /// This function will panic if an unsupported weight type is provided.
    ///
    /// # Examples
    /// ```
    /// use rusty_science::regression::{KNNRegression, WeightType};
    /// let mut knn = KNNRegression::<f64, f64>::new(3);
    /// knn.set_weight_type(WeightType::Distance);
    /// ```
    pub fn set_weight_type(&mut self, weight_type: WeightType) {
        self.weight_type = weight_type;
    }
    

    /// Sets the method that will determine the weights for neighbors, either 'uniform' or 'distance',
    /// and consumes the current instance, returning a new one with the updated weight type.
    /// Builder method, returns self
    ///
    /// # Arguments
    /// * `weight_type` - A `WeightType` enum specifying the desired weighting method.
    ///
    /// # Returns
    /// * `Self` - A new instance of `KNNRegression` with the updated weight type.
    ///
    /// # Examples
    /// ```
    /// use rusty_science::regression::{KNNRegression, WeightType};
    /// let knn = KNNRegression::<f64, f64>::new(3)
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
    /// use rusty_science::regression::{KNNRegression, DistanceMetric};
    /// let mut knn = KNNRegression::<f64, f64>::new(3);
    /// knn.set_distance_metric(DistanceMetric::Manhattan);
    /// ```
    pub fn set_distance_metric(&mut self, distance_metric: DistanceMetric) {
        self.distance_metric = distance_metric;
    }


    /// Modifies the distance metric to the specified type and 
    /// returns the updated `KNNRegression` instance.
    /// Builder method.
    ///
    /// # Arguments
    /// * `distance_metric` - A `DistanceMetric` enum variant specifying the distance metric.
    ///
    /// # Returns
    /// * `Self` - The updated instance of `KNNRegression`.
    ///
    /// # Examples
    /// ```
    /// use rusty_science::regression::{KNNRegression, DistanceMetric};
    /// let knn = KNNRegression::<f64, f64>::new(3)
    ///     .with_distance_metric(DistanceMetric::Manhattan);
    /// ```
    pub fn with_distance_metric(mut self, distance_metric: DistanceMetric) -> Self {
        self.set_distance_metric(distance_metric);
        self
    }

    /// Fits the regression with the training data and labels.
    ///
    /// # Arguments
    /// * `data` - A vector of vectors containing the training data points.
    /// * `labels` - A vector of labels corresponding to the training data points.
    ///
    /// # Examples
    /// ```
    /// use rusty_science::regression::KNNRegression;
    /// let mut knn = KNNRegression::<f64, f64>::new(3);
    /// let data = vec![vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 4.0]];
    /// let labels = vec![0.72, 1.0, 0.26];
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
    /// * An `f64` representing the predicted label for the target data point.
    ///
    /// # Examples
    /// ```
    /// use rusty_science::regression::KNNRegression;
    /// let mut knn = KNNRegression::<f64, f64>::new(3);
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

        let mut weighted_sum = 0.0_f64;
        let mut total_weight = 0.0_f64;

        for neighbor in neighbors.iter() {
            let label = neighbor.label.unwrap();
            let distance = neighbor.distance_to_target;

            let weight = if calculate_distance && distance != 0.0 {
                1.0 / (distance + 1e-8)
            } else {
                1.0
            };

            // Convert label to f64 for calculation
            let label_f64 = label.to_f64().expect("Failed to convert label to f64");

            weighted_sum += weight * label_f64;
            total_weight += weight;
        }

        // Compute weighted average
        let avg_label_f64 = if total_weight == 0.0 {
            let sum_labels: f64 = neighbors
                .iter()
                .map(|neighbor| neighbor.label.unwrap().to_f64().unwrap())
                .sum();
            sum_labels / neighbors.len() as f64
        } else {
            weighted_sum / total_weight
        };

        // Convert the average back to L
        L::from_f64(avg_label_f64).expect("Failed to convert average label from f64")
    }
}

#[cfg(test)]
#[allow(unused_must_use)]
mod tests {
    use approx::assert_relative_eq;
    use super::*;

    #[test]
    fn test_new_knn_regression() {
        let knn = KNNRegression::<f64, f64>::new(3);
        assert_eq!(knn.k, 3);
        assert_eq!(knn.data.len(), 0);
        assert_eq!(knn.labels.len(), 0);
        assert_eq!(knn.weight_type, WeightType::Uniform);
        assert_eq!(knn.distance_metric, DistanceMetric::Euclidean);
    }

    #[test]
    fn test_fit() {
        let mut knn = KNNRegression::<f64, f64>::new(3);
        let data = vec![vec![1.0, 2.0], vec![2.0, 3.0]];
        let labels = vec![1.5, 2.5];
        knn.fit(data.clone(), labels.clone()).expect("Failed to fit KNN model");
        assert_eq!(knn.data, data);
        assert_eq!(knn.labels, labels);
    }

    #[test]
    fn test_set_weights() {
        let mut knn = KNNRegression::<f64, f64>::new(3);
        knn.set_weight_type(WeightType::Distance);
        assert_eq!(knn.weight_type, WeightType::Distance);

        knn.set_weight_type(WeightType::Uniform);
        assert_eq!(knn.weight_type, WeightType::Uniform);
    }
    
    #[test]
    fn test_set_distance_metrics() {
        let mut knn = KNNRegression::<f64, f64>::new(3);
        knn.set_distance_metric(DistanceMetric::Manhattan);
        assert_eq!(knn.distance_metric, DistanceMetric::Manhattan);

        knn.set_distance_metric(DistanceMetric::Euclidean);
        assert_eq!(knn.distance_metric, DistanceMetric::Euclidean);
    }

    #[test]
    fn test_predict() {
        let mut knn = KNNRegression::<f64, f64>::new(3);
        let data = vec![
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![3.0, 3.0],
            vec![4.0, 4.0],
        ];
        let labels = vec![1.0, 2.0, 3.0, 4.0];
        knn.fit(data, labels);

        let target = vec![2.5, 2.5];
        let prediction = knn.predict(target);

        assert!(prediction >= 2.0 && prediction <= 3.0);
    }
    #[test]
    fn test_knn_regression_non_default_weight() {
        let mut knn = KNNRegression::<f64, f64>::new(3);
        let data = vec![
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![3.0, 3.0],
            vec![4.0, 4.0],
        ];
        let labels = vec![1.0, 2.0, 3.0, 4.0];
        knn.fit(data, labels);
        knn.set_weight_type(WeightType::Distance);
        let target = vec![2.5, 2.5];
        let prediction = knn.predict(target);

        assert!(prediction >= 2.0 && prediction <= 3.0);
    }

    #[test]
    #[should_panic]
    fn test_knn_zero_k(){
        let _ = KNNRegression::<f64, f64>::new(0);
    }

    #[test]
    fn set_weight_type() {
        let data = vec![
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![3.0, 3.0],
            vec![4.0, 4.0],
        ];
        let labels = vec![1.0, 2.0, 3.0, 4.0];
        let mut knn = KNNRegression::<f64, f64>::new(3);
        knn.set_weight_type(WeightType::Uniform);
        knn.fit(data, labels);
        knn.predict(vec![1.0, 2.0]);
    }
    
        #[test]
    fn test_empty_training_data() {
        let mut model = KNNRegression::<f64, f64>::new(3);
        let result = model.fit(vec![], vec![]);
        assert!(result.is_err());
        // Verify specific error type if you implement custom errors
        assert!(matches!(result, Err(ModelError::EmptyData)));
    }

    #[test]
    fn test_mismatched_data_label_lengths() {
        let mut model = KNNRegression::<f64, f64>::new(3);
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let labels = vec![10.0]; // One label missing
        let result = model.fit(data, labels);
        assert!(result.is_err());
        assert!(matches!(result, Err(ModelError::DimensionMismatch(_))));
    }

    #[test]
    fn test_inconsistent_feature_dimensions() {
        let mut model = KNNRegression::<f64, f64>::new(3);
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0, 5.0]]; // Second sample has different dimension
        let labels = vec![10.0, 20.0];
        let result = model.fit(data, labels);
        assert!(result.is_err());
        assert!(matches!(result, Err(ModelError::DimensionMismatch(_))));
    }

    #[test]
    fn test_single_training_example() {
        let mut model = KNNRegression::<f64, f64>::new(1);
        let data = vec![vec![1.0, 2.0]];
        let labels = vec![10.0];
        model.fit(data.clone(), labels.clone()).unwrap();
        
        let prediction = model.predict(data[0].clone());
        assert_relative_eq!(prediction, labels[0]);
    }
    
    #[test]
    #[should_panic]
    fn test_predict_with_invalid_dimensions() {
        let mut model = KNNRegression::<f64, f64>::new(3);
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let labels = vec![10.0, 20.0, 30.0];
        model.fit(data, labels).unwrap();
        
        // Testing with wrong number of features 
        let _ = model.predict(vec![1.0, 2.0, 3.0]); // 3 features instead of 2
    }

    #[test]
    fn test_k_larger_than_training_set() {
        let mut model = KNNRegression::<f64, f64>::new(10); // k=10
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]; // Only 3 examples
        let labels = vec![10.0, 20.0, 30.0];
        model.fit(data.clone(), labels.clone()).unwrap();
        
        // Should still work but effectively use k=3
        let prediction = model.predict(vec![2.0, 3.0]);
        
        // Verify prediction is reasonable (in the range of training labels)
        assert!(prediction >= 10.0 && prediction <= 30.0);
    }
    
    #[test]
    fn test_extrapolation_behavior() {
        let mut model = KNNRegression::<f64, f64>::new(3);
        let data = vec![vec![1.0, 1.0], vec![2.0, 2.0], vec![3.0, 3.0]];
        let labels = vec![1.0, 2.0, 3.0];
        model.fit(data, labels).unwrap();
        
        // Test prediction for a point outside the training range
        let prediction = model.predict(vec![10.0, 10.0]);
        // Should predict the closest point's label or a weighted average
        assert!(prediction > 0.0);
    }
    
    #[test]
    fn test_different_weight_types() {
        let data = vec![vec![1.0, 1.0], vec![5.0, 5.0], vec![10.0, 10.0]];
        let labels = vec![1.0, 5.0, 10.0];
        let test_point = vec![6.0, 6.0];
        
        // Test uniform weights
        let mut uniform_model = KNNRegression::<f64, f64>::new(2);
        uniform_model.set_weight_type(WeightType::Uniform);
        uniform_model.fit(data.clone(), labels.clone()).unwrap();
        let uniform_pred = uniform_model.predict(test_point.clone());
        
        // Test distance weights
        let mut distance_model = KNNRegression::<f64, f64>::new(2);
        distance_model.set_weight_type(WeightType::Distance);
        distance_model.fit(data, labels).unwrap();
        let distance_pred = distance_model.predict(test_point);
        
        // Different weight types should generally give different predictions
        assert!(uniform_pred != distance_pred);
    }
    
    #[test]
    fn test_integer_data_types() {
        let mut model = KNNRegression::<i32, i32>::new(3);
        let data = vec![vec![1, 2], vec![3, 4], vec![5, 6]];
        let labels = vec![10, 20, 30];
        model.fit(data, labels).unwrap();
        
        let prediction = model.predict(vec![2, 3]);
        assert!(prediction >= 10 && prediction <= 30);
    }
    
    #[test]
    fn test_mixed_data_types() {
        let mut model = KNNRegression::<f64, i32>::new(3);
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let labels = vec![10, 20, 30];
        model.fit(data, labels);
        
        let prediction = model.predict(vec![2.0, 3.0]);
        assert!(prediction >= 10 && prediction <= 30);
    }
    
    #[test]
    fn test_builder_pattern() {
        let model = KNNRegression::<f64, f64>::new(3)
            .with_weight_type(WeightType::Distance)
            .with_distance_metric(DistanceMetric::Euclidean);
            
        assert_eq!(model.k, 3);
        assert_eq!(model.weight_type, WeightType::Distance);
        assert_eq!(model.distance_metric, DistanceMetric::Euclidean);
    }
    
    #[test]
    fn test_default() {
        let model = KNNRegression::<f64, f64>::default();
        assert_eq!(model.k, 5); // Assuming default k is 5
        assert_eq!(model.weight_type, WeightType::Uniform); 
        assert_eq!(model.distance_metric, DistanceMetric::Euclidean);
    }
    
    #[test]
    fn test_exact_match() {
        let mut model = KNNRegression::<f64, f64>::new(1);
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let labels = vec![10.0, 20.0, 30.0];
        model.fit(data.clone(), labels.clone()).unwrap();
        
        // Test prediction on exact match to training point
        for i in 0..data.len() {
            let prediction = model.predict(data[i].clone());
            assert_relative_eq!(prediction, labels[i]);
        }
    }
    
    #[test]
    fn test_clone() {
        let mut original = KNNRegression::<f64, f64>::new(3);
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let labels = vec![10.0, 20.0, 30.0];
        original.fit(data, labels).unwrap();
        
        // Clone the model
        let cloned = original.clone();
        
        // Both should give the same prediction
        let test_point = vec![2.0, 3.0];
        let original_pred = original.predict(test_point.clone());
        let cloned_pred = cloned.predict(test_point);
        
        assert_relative_eq!(original_pred, cloned_pred);
    }

}


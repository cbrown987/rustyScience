use crate::common::knn::neighbors;

pub struct KNNRegression {
    k: usize,
    data: Vec<Vec<f64>>,
    labels: Vec<f64>,
    weight_type: String,
    distance_metric: String,
}

impl KNNRegression {
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
    /// use rustyScience::regression::KNNRegression;
    /// let knn = KNNRegression::new(3);
    /// ```
    pub fn new(k: usize) -> Self {
        if k <= 0 {
            panic!("K cannot be zero");
        }
        Self {
            k,
            data: vec![],
            labels: vec![],
            weight_type: "uniform".to_string(),
            distance_metric: "euclidean".to_string(),
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
    /// use rustyScience::regression::KNNRegression;
    /// let mut knn = KNNRegression::new(3);
    /// knn.set_weight_type("uniform".to_string());
    /// ```
    pub fn set_weight_type(&mut self, weight_type: String) {
        if weight_type.to_lowercase() == "uniform" || weight_type.to_lowercase() == "distance" {
            self.weight_type = weight_type;
        } else {
            panic!("Unsupported or unknown weight type, use uniform or distance");
        }
    }

    /// Sets the distance metric to be used for finding neighbors.
    ///
    /// # Arguments
    /// * `distance_metric` - A string specifying the distance metric, e.g., 'euclidean' or 'manhattan'.
    ///
    /// # Examples
    /// ```
    /// use rustyScience::regression::KNNRegression;
    /// let mut knn = KNNRegression::new(3);
    /// knn.set_distance_metrics("manhattan".to_string());
    /// ```
    pub fn set_distance_metrics(&mut self, distance_metric: String) {
        self.distance_metric = distance_metric;
    }

    /// Fits the regression with the training data and labels.
    ///
    /// # Arguments
    /// * `data` - A vector of vectors containing the training data points.
    /// * `labels` - A vector of labels corresponding to the training data points.
    ///
    /// # Examples
    /// ```
    /// use rustyScience::regression::KNNRegression;
    /// let mut knn = KNNRegression::new(3);
    /// let data = vec![vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 4.0]];
    /// let labels = vec![0.72, 1.0, 0.26];
    /// knn.fit(data, labels);
    /// ```
    pub fn fit(&mut self, data: Vec<Vec<f64>>, labels: Vec<f64>) {
        self.data = data;
        self.labels = labels
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
    /// use rustyScience::regression::KNNRegression;
    /// let mut knn = KNNRegression::new(3);
    /// let data = vec![vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 4.0]];
    /// let labels = vec![0.72, 1.0, 0.26];
    /// knn.fit(data, labels);
    /// let prediction = knn.predict(vec![2.5, 3.5]);
    /// println!("Predicted label: {}", prediction);
    /// ```
    pub fn predict(&self, target: Vec<f64>) -> f64 {
        self._predict(target)
    }

    fn _predict(&self, target: Vec<f64>) -> f64 {
        let calculate_distance = self.weight_type == "distance";

        let neighbors = neighbors(
            self.data.clone(), None, Option::from(self.labels.clone()), Option::from(target), self.k,
            self.distance_metric.clone(), calculate_distance);


        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;

        for neighbor in &neighbors {
            let label = *neighbor.last().unwrap();
            let distance = neighbor[neighbor.len() - 2];

            let weight = if calculate_distance && distance != 0.0 {
                1.0 / distance
            } else {
                1.0
            };

            weighted_sum += weight * label;
            total_weight += weight;
        }

        // Compute weighted average
        if total_weight == 0.0 {
            neighbors.iter().map(|neighbor| *neighbor.last().unwrap()).sum::<f64>() / neighbors.len() as f64
        } else {
            weighted_sum / total_weight
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_knn_regression() {
        let knn = KNNRegression::new(3);
        assert_eq!(knn.k, 3);
        assert_eq!(knn.data.len(), 0);
        assert_eq!(knn.labels.len(), 0);
        assert_eq!(knn.weight_type, "uniform");
        assert_eq!(knn.distance_metric, "euclidean");
    }

    #[test]
    fn test_fit() {
        let mut knn = KNNRegression::new(3);
        let data = vec![vec![1.0, 2.0], vec![2.0, 3.0]];
        let labels = vec![1.5, 2.5];
        knn.fit(data.clone(), labels.clone());
        assert_eq!(knn.data, data);
        assert_eq!(knn.labels, labels);
    }

    #[test]
    fn test_set_weights() {
        let mut knn = KNNRegression::new(3);
        knn.set_weight_type("distance".to_string());
        assert_eq!(knn.weight_type, "distance");

        knn.set_weight_type("uniform".to_string());
        assert_eq!(knn.weight_type, "uniform");
    }

    #[should_panic]
    #[test]
    fn test_unsupported_weights() {
        let mut knn = KNNRegression::new(3);
        knn.set_weight_type("unsupported".to_string());
    }

    #[test]
    fn test_set_distance_metrics() {
        let mut knn = KNNRegression::new(3);
        knn.set_distance_metrics("manhattan".to_string());
        assert_eq!(knn.distance_metric, "manhattan");

        knn.set_distance_metrics("euclidean".to_string());
        assert_eq!(knn.distance_metric, "euclidean");
    }

    #[test]
    fn test_predict() {
        let mut knn = KNNRegression::new(3);
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
        let mut knn = KNNRegression::new(3);
        let data = vec![
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![3.0, 3.0],
            vec![4.0, 4.0],
        ];
        let labels = vec![1.0, 2.0, 3.0, 4.0];
        knn.fit(data, labels);
        knn.set_weight_type(String::from("uniform"));
        let target = vec![2.5, 2.5];
        let prediction = knn.predict(target);

        assert!(prediction >= 2.0 && prediction <= 3.0);
    }
    
    #[test]
    #[should_panic]
    fn test_knn_zero_k(){
        let _ = KNNRegression::new(0);
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
        let mut knn = KNNRegression::new(3);
        knn.set_weight_type("distance".to_string());
        knn.fit(data, labels);
        knn.predict(vec![1.0, 2.0]);
    }
}


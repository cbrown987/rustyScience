use crate::common::knn::neighbors;

pub struct KNNRegression {
    k: usize,
    data: Vec<Vec<f64>>, 
    labels: Vec<f64>,
    weight_type: String,
    distance_metric: String
}

impl KNNRegression {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            data: vec![],
            labels: vec![],
            weight_type: "uniform".to_string(),
            distance_metric: "euclidean".to_string(),
        }
    }

    pub fn fit(&mut self, data: Vec<Vec<f64>>, labels: Vec<f64>) {
        self.data = data; 
        self.labels = labels
    }

    // Sets the method that will determine the weights, uniform or distance
    fn set_weights(&mut self, weight_type: String) {
        if weight_type.to_lowercase() == "uniform" || weight_type.to_lowercase() == "distance" {
            self.weight_type = weight_type;
        } else { panic!("Unsupported or unknown weight type, use uniform or distance"); }
    }
    
    // Sets the distance algorithm
    fn set_distance_metrics(&mut self, distance_metric: String) {
        self.distance_metric = distance_metric;
    }
    fn _predict(&self, target: Vec<f64>) -> f64 {
        let mut calculate_distance: bool = false;
        
        if self.weight_type == "distance" {
            calculate_distance = true;
        }

        let neighbors = neighbors(
            self.data.clone(), None, Option::from(self.labels.clone()), Option::from(target), self.k,
            self.distance_metric.clone(), calculate_distance);
        
        
        let sum: f64 = neighbors
            .iter()
            .map(|neighbor| *neighbor.last().unwrap())  // Get the label (last element)
            .sum();

        let prediction = sum / neighbors.len() as f64;

        prediction
    }
    pub fn predict(&self, target: Vec<f64>) -> f64 {
        self._predict(target)
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
        knn.set_weights("distance".to_string());
        assert_eq!(knn.weight_type, "distance");

        knn.set_weights("uniform".to_string());
        assert_eq!(knn.weight_type, "uniform");
    }
    
    #[should_panic]
    #[test]
    fn test_unsupported_weights() {
        let mut knn = KNNRegression::new(3);
        knn.set_weights("unsupported".to_string());
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
}

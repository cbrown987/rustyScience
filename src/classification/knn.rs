use crate::common::knn::{neighbors, Neighbor};
use std::hash::Hash;
use num::Num;
use num_traits::ToPrimitive;

pub struct KNNClassifier<D, L> {
    k: usize,
    data: Vec<Vec<D>>,
    labels: Vec<L>,
    distance_metric: String,
    weight_type: String,
}

impl<D, L> KNNClassifier<D, L>
where
    D: Num + Copy + Clone + PartialOrd + ToPrimitive,
    L: Copy + Clone + Eq + Hash,
{
    pub fn new(k: usize) -> Self {
        if k == 0 {
            panic!("K cannot be zero");
        }
        Self {
            k,
            data: vec![],
            labels: vec![],
            distance_metric: "euclidean".to_string(),
            weight_type: "distance".to_string(),
        }
    }

    pub fn set_weight_type(&mut self, weight_type: String) {
        if weight_type.to_lowercase() == "uniform" || weight_type.to_lowercase() == "distance" {
            self.weight_type = weight_type;
        } else {
            panic!("Unsupported or unknown weight type, use 'uniform' or 'distance'");
        }
    }

    pub fn set_distance_metrics(&mut self, distance_metric: String) {
        self.distance_metric = distance_metric;
    }

    pub fn fit(&mut self, data: Vec<Vec<D>>, labels: Vec<L>) {
        self.data = data;
        self.labels = labels;
    }

    pub fn predict(&self, target: Vec<D>) -> L {
        self._predict(target)
    }

    fn _predict(&self, target: Vec<D>) -> L {
        let calculate_distance = self.weight_type.to_lowercase() == "distance";
        let neighbors: Vec<Neighbor<D, L>> = neighbors(
            self.data.clone(),
            Some(self.labels.clone()),
            Some(target),
            self.k,
            self.distance_metric.clone(),
            calculate_distance,
        );

        let mut label_weights = std::collections::HashMap::new();

        for neighbor in neighbors.iter() {
            if let Some(label) = neighbor.label {
                let distance = neighbor.distance_to_target;
                let weight = if calculate_distance {
                    1.0 / (distance + 1e-8)
                } else {
                    1.0
                };
                *label_weights.entry(label).or_insert(0.0) += weight;
            }
        }

        // Get the label with the highest weighted count
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
        knn.set_weight_type("uniform".to_string());
        knn.fit(data, labels);

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
        knn.set_weight_type("distance".to_string());
        knn.fit(data, labels);

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
        knn.fit(data, labels);

        // Predict label for a point exactly at [3.0, 3.0]
        let target = vec![3.0, 3.0];
        let predicted_label = knn.predict(target);

        // Expected label is 1
        assert_eq!(predicted_label, 1);
    }

    #[test]
    #[should_panic(expected = "Unsupported or unknown weight type")]
    fn test_knnclassifier_invalid_weight_type() {
        let mut knn = KNNClassifier::<f64, i32>::new(3);
        knn.set_weight_type("invalid_weight".to_string());
    }

    #[test]
    #[should_panic(expected = "K cannot be zero")]
    fn test_knnclassifier_k_zero() {
        KNNClassifier::<f64, i32>::new(0);
    }

    #[test]
    #[should_panic(expected = "Unknown distance metric")]
    fn test_knnclassifier_invalid_distance_metric() {
        let mut knn = KNNClassifier::<f64, i32>::new(3);
        knn.set_distance_metrics("invalid_metric".to_string());
        knn.fit(vec![vec![1.0, 2.0]], vec![0]);
        knn.predict(vec![1.0, 2.0]);
    }
}

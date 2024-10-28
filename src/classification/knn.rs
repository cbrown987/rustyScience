use crate::common::knn::neighbors;
use std::collections::HashMap;

pub struct KNNClassifier {
    k: usize,
    data: Vec<Vec<f64>>,
    labels: Vec<i64>,
    distance_metric: String,
    weight_type: String
}

impl KNNClassifier {

    pub fn new(k: usize) -> Self{
        if k <= 0 {
            panic!("K cannot be zero");
        }
        Self{
            k,
            data: vec![vec![]],
            labels: vec![],
            distance_metric: "euclidean".to_string(),
            weight_type: "distance".to_string(),
        }
    }
    // Sets the method that will determine the weights, uniform or distance
    pub fn set_weights(&mut self, weight_type: String) {
        if weight_type.to_lowercase() == "uniform" || weight_type.to_lowercase() == "distance" {
            self.weight_type = weight_type;
        } else { panic!("Unsupported or unknown weight type, use uniform or distance"); }
    }
    // Sets the distance algoritum
    pub fn set_distance_metrics(&mut self, distance_metric: String) {
        self.distance_metric = distance_metric;
    }
    pub fn fit(&mut self, data: Vec<Vec<f64>>, labels: Vec<i64>){
        self.data = data;
        self.labels = labels;
    }
    fn _predict(&self, target: Vec<f64>) -> i64{
        let neighbors = neighbors(
            self.data.clone(), Option::from(self.labels.clone()), None, Option::from(target), self.k,
            self.distance_metric.clone(), false);

        let last_elements: Vec<i64> = neighbors
            .iter()
            .filter_map(|v| v.last())
            .map(|&x| x as i64)
            .collect();

        let mut counts: HashMap<i64, usize> = HashMap::new();
        for &element in &last_elements {
            *counts.entry(element).or_insert(0) += 1;
        }

        let (most_frequent_element, _) = counts
            .iter()
            .max_by_key(|&(_, count)| count)
            .expect("List should not be empty");
        most_frequent_element.clone()
    }
    pub fn predict(&self, target: Vec<f64>) -> i64 {
        self._predict(target)
    }


}

#[cfg(test)]
mod tests {
    use crate::classification::knn::KNNClassifier;
    use crate::common::test_utils::create_data_labeled;

    #[test]
    fn test_knn_classifier_small_dataset(){
        if let Some(dataset) = create_data_labeled().get("small_data") {
            let target = vec![7.0,8.0];
            let mut knn = KNNClassifier::new( 3);
            knn.fit(dataset.data.clone(), dataset.labels.clone());
            let prediction = knn.predict(target);

            assert_eq!(prediction, 1);

            let target = vec![2.0, 2.0];
            let mut knn = KNNClassifier::new( 1);
            knn.fit(dataset.data.clone(), dataset.labels.clone());
            let prediction = knn.predict(target);

            assert_eq!(prediction, 10);
        }

    }

}
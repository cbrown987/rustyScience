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
    pub fn set_weight_type(&mut self, weight_type: String) {
        if weight_type.to_lowercase() == "uniform" || weight_type.to_lowercase() == "distance" {
            self.weight_type = weight_type;
        } else { panic!("Unsupported or unknown weight type, use uniform or distance"); }
    }
    // Sets the distance algorithm
    pub fn set_distance_metrics(&mut self, distance_metric: String) {
        self.distance_metric = distance_metric;
    }
    
    pub fn fit(&mut self, data: Vec<Vec<f64>>, labels: Vec<i64>){
        self.data = data;
        self.labels = labels;
    }
    
    fn _predict(&self, target: Vec<f64>) -> i64{
        let calculate_distance = self.weight_type == "distance";
        
        let neighbors = neighbors(
            self.data.clone(), Option::from(self.labels.clone()), None, Option::from(target), self.k,
            self.distance_metric.clone(), calculate_distance);

        let mut weighted_counts: HashMap<i64, f64> = HashMap::new();

        for neighbor in neighbors.iter() {
            let label = *neighbor.last().expect("Neighbor should have a label") as i64;
            let distance = *neighbor
                .get(neighbor.len().wrapping_sub(2))
                .expect("Neighbor should have a distance");

            let mut weight = 1.0;
            if calculate_distance {
                weight = 1.0 / distance + 1e-8;
            }
            *weighted_counts.entry(label).or_insert(0.0) += weight;
        }

        // Get the label with the highest weighted count
        weighted_counts
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(label, _)| label)
            .expect("Neighbor list should not be empty")
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
    
    #[test]
    fn test_knn_classifier_non_default_weight(){
        if let Some(dataset) = create_data_labeled().get("small_data") {
            let target = vec![7.0,8.0];
            let mut knn = KNNClassifier::new( 3);
            knn.set_weight_type("uniform".parse().unwrap());
            knn.fit(dataset.data.clone(), dataset.labels.clone());
            let prediction = knn.predict(target);

            assert_eq!(prediction, 1);

            let target = vec![2.0, 2.0];
            let mut knn = KNNClassifier::new( 1);
            knn.set_weight_type("uniform".parse().unwrap());
            knn.fit(dataset.data.clone(), dataset.labels.clone());
            let prediction = knn.predict(target);

            assert_eq!(prediction, 10);
        }
    }
    
    #[test]
    fn test_knn_classifier_non_default_distance(){
        if let Some(dataset) = create_data_labeled().get("small_data") {
            let target = vec![7.0,8.0];
            let mut knn = KNNClassifier::new( 3);
            knn.set_distance_metrics("euclidean".parse().unwrap());
            knn.fit(dataset.data.clone(), dataset.labels.clone());
            let prediction = knn.predict(target);

            assert_eq!(prediction, 1);

            let target = vec![2.0, 2.0];
            let mut knn = KNNClassifier::new( 1);
            knn.set_distance_metrics("euclidean".parse().unwrap());
            knn.fit(dataset.data.clone(), dataset.labels.clone());
            let prediction = knn.predict(target);

            assert_eq!(prediction, 10);
        }
    }
}
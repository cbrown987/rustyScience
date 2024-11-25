use crate::common::knn::{neighbors, Neighbor};
use num::Num;
use num_traits::ToPrimitive;

#[cfg(feature = "graphing")]
use plotters::prelude::*;

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
            distance_metric: "euclidean".to_string(),
            weight_type: "distance".to_string(),
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
    /// use rusty_science::classification::KNNClassifier;
    /// let mut knn = KNNClassifier::<f64, i64>::new(3);
    /// knn.set_weight_type("uniform".to_string());
    /// ```
    pub fn set_weight_type(&mut self, weight_type: String) {
        if weight_type.to_lowercase() == "uniform" || weight_type.to_lowercase() == "distance" {
            self.weight_type = weight_type;
        } else {
            panic!("Unsupported or unknown weight type, use 'uniform' or 'distance'");
        }
    }

    /// Sets the distance metric to be used for finding neighbors.
    ///
    /// # Arguments
    /// * `distance_metric` - A string specifying the distance metric, e.g., 'euclidean' or 'manhattan'.
    ///
    /// # Examples
    /// ```
    /// use rusty_science::classification::KNNClassifier;
    /// let mut knn = KNNClassifier::<f64, i64>::new(3);
    /// knn.set_distance_metrics("manhattan".to_string());
    /// ```
    pub fn set_distance_metrics(&mut self, distance_metric: String) {
        self.distance_metric = distance_metric;
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
    pub fn fit(&mut self, data: Vec<Vec<D>>, labels: Vec<L>) {
        self._fit(data, labels);
    }
    
    fn _fit(&mut self, data: Vec<Vec<D>>, labels: Vec<L>){
        self.data = data;
        self.labels = labels;
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
    #[cfg(feature = "graphing")]
    pub fn graph_data_to_image(&self, title: &str, path: &str){
        let drawing_area = BitMapBackend::new(path, (800, 600))
            .into_drawing_area();
        drawing_area.fill(&WHITE).unwrap();

        let (x_min, x_max) = self.data.iter().flat_map(|v| v.get(0)).fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &val| {
            let val = val.to_f64().unwrap();
            (min.min(val), max.max(val))
        });
        let (y_min, y_max) = self.data.iter().flat_map(|v| v.get(1)).fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &val| {
            let val = val.to_f64().unwrap();
            (min.min(val), max.max(val))
        });

        // Create the chart.
        let mut chart = ChartBuilder::on(&drawing_area)
            .caption(title, ("sans-serif", 20).into_font())
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(x_min..x_max, y_min..y_max)
            .unwrap();

        chart.configure_mesh()
            .x_desc("Feature 1")
            .y_desc("Feature 2")
            .draw()
            .unwrap();

        // Plot each data point, color-coded by label.
        for (point, &label) in self.data.iter().zip(self.labels.iter()) {
            if let (Some(&x), Some(&y)) = (point.get(0), point.get(1)) {
                let x = x.to_f64().unwrap();
                let y = y.to_f64().unwrap();
                let color = if label.to_f64().unwrap() > 0.5 {
                    &BLUE
                } else {
                    &RED
                };

                chart
                    .draw_series(PointSeries::of_element(
                        [(x, y)],
                        5,
                        color,
                        &|c, s, st| {
                            EmptyElement::at(c)
                                + Circle::new((0, 0), s, st.filled())
                        },
                    ))
                    .unwrap();
            }
        }

        drawing_area.present().expect("Failed to save the chart as an image");
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
    
    #[test]
    #[cfg(feature = "graphing")]
    fn test_knnclassifier_graph_data() {
        let mut knn = KNNClassifier::<f64, i32>::new(3);
        let data = vec![
            vec![1.0, 2.0],
            vec![2.0, 3.0],
            vec![3.0, 3.0],
            vec![4.0, 4.0],
        ];
        let labels = vec![0, 0, 1, 4];
        knn.fit(data, labels);
        knn.graph_data_to_image("Title", "knn_plot.png")
        
    }
}

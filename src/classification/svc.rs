use std::fmt::Debug;
use num_traits::{Num, NumCast, ToPrimitive};

pub struct BinarySVC<D, L> {
    data: Vec<Vec<D>>,
    weights: Vec<f64>,
    bias: f64,
    label_neg: Option<L>,
    label_pos: Option<L>,
    epochs: usize,
    learning_rate: f64,
    regularization_factor: f64,
}

impl<D, L> BinarySVC<D, L>
where
    D: Num + Copy + Clone + PartialOrd + ToPrimitive,
    L: Num + Copy + Clone + PartialOrd + ToPrimitive + Debug + NumCast + std::fmt::Display,
{
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            weights: Vec::new(),
            bias: 0.0,
            label_neg: None,
            label_pos: None,
            epochs: 1000,
            learning_rate: 0.001,
            regularization_factor: 1.0,
        }
    }
    pub fn set_epochs(&mut self, epochs: usize) {
        self.epochs = epochs;
    }
    pub fn set_learning_rate(&mut self, learning_rate: f64) {
        self.learning_rate = learning_rate;
    }
    pub fn set_regularization_factor(&mut self, regularization_factor: f64) {
        self.regularization_factor = regularization_factor;
    }
    pub fn fit(&mut self, data: Vec<Vec<D>>, labels: Vec<L>) {
        self._fit(data, labels);
    }

    fn _fit(&mut self, data: Vec<Vec<D>>, labels: Vec<L>) {
        self.data = data;

        // Identify unique labels
        let mut unique_labels = labels.iter().cloned().collect::<Vec<L>>();
        unique_labels.sort_by(|a, b| format!("{:?}", a).cmp(&format!("{:?}", b)));
        unique_labels.dedup();
        if unique_labels.len() != 2 {
            panic!("Binary SVC supports only binary classification.");
        }

        // Assign labels to internal variables
        self.label_neg = Some(unique_labels[0].clone());
        self.label_pos = Some(unique_labels[1].clone());

        // Map labels to internal representations as f64
        let internal_labels: Vec<f64> = labels
            .iter()
            .map(|label| {
                if *label == *self.label_neg.as_ref().unwrap() {
                    -1.0
                } else if *label == *self.label_pos.as_ref().unwrap() {
                    1.0
                } else {
                    panic!("Unexpected label found.");
                }
            })
            .collect();

        let n_samples = self.data.len();
        let n_features = self.data[0].len();

        // Initialize weights and bias
        self.weights = vec![0.0; n_features];
        self.bias = 0.0;

        let eta = self.learning_rate;
        let c = self.regularization_factor;

        for _ in 0..self.epochs {
            for i in 0..n_samples {
                let xi = &self.data[i];
                let yi = internal_labels[i];

                // Compute the prediction
                let mut wx = 0.0;
                for j in 0..n_features {
                    wx += self.weights[j] * xi[j].to_f64().unwrap();
                }
                wx += self.bias;

                // Check if the sample is misclassified
                if yi * wx < 1.0 {
                    // Misclassified or within margin
                    for j in 0..n_features {
                        self.weights[j] -= eta * (self.weights[j] - c * yi * xi[j].to_f64().unwrap());
                    }
                    self.bias += eta * c * yi;
                } else {
                    // Correctly classified
                    for j in 0..n_features {
                        self.weights[j] -= eta * self.weights[j];
                    }
                    // Bias remains the same
                }
            }
        }
    }


    pub fn predict(&self, target: Vec<D>) -> L {
        self._predict(target)
    }

    fn _predict(&self, target: Vec<D>) -> L {
        let mut result = self.bias;
        for (wi, xi) in self.weights.iter().zip(target.iter()) {
            result += wi * xi.to_f64().unwrap();
        }
        if result >= 0.0 {
            self.label_pos.as_ref().unwrap().clone()
        } else {
            self.label_neg.as_ref().unwrap().clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let svc: BinarySVC<f64, i32> = BinarySVC::new();
        assert_eq!(svc.epochs, 1000);
        assert_eq!(svc.learning_rate, 0.001);
        assert_eq!(svc.regularization_factor, 1.0);
        assert!(svc.data.is_empty());
        assert!(svc.weights.is_empty());
    }

    #[test]
    fn test_set_epochs() {
        let mut svc: BinarySVC<f64, i32> = BinarySVC::new();
        svc.set_epochs(500);
        assert_eq!(svc.epochs, 500);
    }

    #[test]
    fn test_set_learning_rate() {
        let mut svc: BinarySVC<f64, i32> = BinarySVC::new();
        svc.set_learning_rate(0.01);
        assert_eq!(svc.learning_rate, 0.01);
    }

    #[test]
    fn test_set_regularization_factor() {
        let mut svc: BinarySVC<f64, i32> = BinarySVC::new();
        svc.set_regularization_factor(0.5);
        assert_eq!(svc.regularization_factor, 0.5);
    }

    #[test]
    fn test_fit_binary_classification() {
        let mut svc: BinarySVC<f64, i32> = BinarySVC::new();

        let data = vec![
            vec![2.0, 3.0],
            vec![1.0, 1.0],
            vec![4.0, 5.0],
            vec![1.0, 0.0],
        ];
        let labels = vec![1, 1, -1, -1];

        svc.fit(data.clone(), labels.clone());

        // Verify internal state
        assert_eq!(svc.data, data);
        assert!(svc.weights.len() == 2); // Two features
        assert!(svc.label_neg.is_some());
        assert!(svc.label_pos.is_some());
        assert_eq!(svc.label_neg.unwrap(), -1);
        assert_eq!(svc.label_pos.unwrap(), 1);
    }

    #[test]
    fn test_predict() {
        let mut svc: BinarySVC<f64, i32> = BinarySVC::new();
        svc.set_learning_rate(0.01); // Increased learning rate
        svc.set_epochs(5000);        // Increased number of epochs

        let data = vec![
            vec![2.0, 3.0],
            vec![1.0, 1.0],
            vec![4.0, 5.0],
            vec![1.0, 0.0],
        ];
        let labels = vec![1, 1, -1, -1];

        svc.fit(data.clone(), labels.clone());

        // Test predictions
        assert_eq!(svc.predict(vec![2.0, 3.0]), 1);
        assert_eq!(svc.predict(vec![1.0, 1.0]), 1);
        assert_eq!(svc.predict(vec![4.0, 5.0]), -1);
        assert_eq!(svc.predict(vec![1.0, 0.0]), 1);
    }

    #[test]
    #[should_panic(expected = "Binary SVC supports only binary classification.")]
    fn test_fit_non_binary_labels() {
        let mut svc: BinarySVC<f64, i32> = BinarySVC::new();

        let data = vec![
            vec![2.0, 3.0],
            vec![1.0, 1.0],
            vec![4.0, 5.0],
        ];
        let labels = vec![1, 0, -1]; // Non-binary labels

        svc.fit(data, labels); // Should panic
    }
}


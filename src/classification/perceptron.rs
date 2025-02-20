use num::Integer;
use num_traits::{Num, NumCast, ToPrimitive};
use crate::common::utils::shuffle_data_labels;
use crate::{panic_labels_not_binary, panic_untrained};

/// A simple multi-class perceptron structure.
///
/// # Generics
/// * `D` - data type, must be numeric.
/// * `L` - label type, must be integer-based.
pub struct MultiClassPerceptron<D, L> {
    model_name: String,
    pub(crate) penalty: String,
    pub(crate) alpha: f32,
    pub(crate) shuffle: bool,

    data: Vec<Vec<D>>,     // 2D data (samples, features)
    labels: Vec<L>,        // One label per data row
    distinct_labels: Vec<L>, // Unique set of possible labels

    // Weights is 2D: (num_classes , num_features)
    // For each class, we have a weight vector for the features.
    weights: Vec<Vec<f64>>,

    // One bias per class
    biases: Vec<f64>,

    // Training hyperparameters
    learning_rate: f64,
    epochs: usize,
}


impl<D, L> MultiClassPerceptron<D, L>
where
    D: Num + Copy + Clone + PartialOrd + ToPrimitive + NumCast,
    L: Num + Integer + Copy + Clone + NumCast + PartialEq,
{
    /// Create a new perceptron with default settings.
    pub fn new() -> Self {
        Self {
            model_name: "MultiClassPerceptron".to_string(),
            penalty: "".to_string(),
            alpha: 0.0,
            shuffle: false,
            data: vec![],
            labels: vec![],
            distinct_labels: vec![],
            weights: vec![],
            biases: vec![],
            learning_rate: 0.3,
            epochs: 10,
        }
    }

    pub fn set_penalty(&mut self, penalty: String) {
        self.penalty = penalty;
    }
    pub fn set_alpha(&mut self, alpha: f32) {
        self.alpha = alpha;
    }
    pub fn set_shuffle(&mut self, shuffle: bool) {
        self.shuffle = shuffle;
    }
    pub fn set_epochs(&mut self, epochs: usize) {
        self.epochs = epochs;
    }
    pub fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr;
    }

    /// Fit the perceptron with training data and labels.
    ///
    /// `data` is a vector of vectors (each inner vec is feature set of a sample).
    /// `labels` is a vector of integer-type labels.
    pub fn fit(&mut self, data: Vec<Vec<D>>, labels: Vec<L>) {
        self.data = data;
        self.labels = labels;

        let mut unique = self.labels.clone();
        unique.sort();
        unique.dedup();
        self.distinct_labels = unique;

        if self.data.is_empty() {
            return;
        }

        let num_features = self.data[0].len();
        let num_classes = self.distinct_labels.len();

        self.weights = vec![vec![0.0; num_features]; num_classes];
        self.biases = vec![0.0; num_classes];
        
        self._fit()
    }
    
    fn _fit(&mut self) {
        for _ in 0..self.epochs {
            if self.shuffle {
                shuffle_data_labels(&mut self.data, &mut self.labels);
            }

            for (row, label) in self.data.iter().zip(self.labels.iter()) {
                let features_f64: Vec<f64> = row.iter().map(|x| x.to_f64().unwrap_or(0.0)).collect();
                let pred_class_index =  perceptron_predict_idx(&features_f64, &self.weights, &self.biases);

                let actual_class_index = self
                    .distinct_labels
                    .iter()
                    .position(|&l| l == *label)
                    .unwrap();

                if pred_class_index != actual_class_index {
                    update_weights_biases(&mut self.weights, &mut self.biases, self.learning_rate, features_f64, actual_class_index, pred_class_index)
                }
            }
        }
    }

    /// Predict the label for a single data sample.
    pub fn predict(&self, sample: Vec<D>) -> L {
        panic_untrained!(self.labels.len() == 0, self.model_name);

        let features_f64: Vec<f64> = sample
            .into_iter()
            .map(|x| x.to_f64().unwrap_or(0.0))
            .collect();

        let class_idx = perceptron_predict_idx(&features_f64, &self.weights, &self.biases);
        self.distinct_labels[class_idx]
    }
}

pub struct BinaryPerceptron<D, L> {
    model_name: String,
    pub(crate) penalty: String,
    pub(crate) alpha: f32,
    pub(crate) shuffle: bool,

    data: Vec<Vec<D>>,     // 2D data (samples, features)
    labels: Vec<L>,

    // Weights is 2D: (num_classes , num_features)
    // For each class, we have a weight vector for the features.
    weights: Vec<Vec<f64>>,

    // One bias per class
    biases: Vec<f64>,

    // Training hyperparameters
    learning_rate: f64,
    epochs: usize,
}

impl<D, L> BinaryPerceptron<D, L> 
where
    D: Num + Copy + Clone + PartialOrd + ToPrimitive + NumCast,
    L: Num + Integer + Copy + Clone + NumCast + PartialEq + Into<i64>

{
    pub fn new() -> Self {
        Self {
            model_name: "BinaryPerceptron".to_string(),
            penalty: "".to_string(),
            alpha: 0.0,
            shuffle: false,
            data: vec![],
            labels: vec![],
            weights: vec![],
            biases: vec![],
            learning_rate: 0.3,
            epochs: 10,
        }
    }
    pub fn set_penalty(&mut self, penalty: String) {
        self.penalty = penalty;
    }
    pub fn set_alpha(&mut self, alpha: f32) {
        self.alpha = alpha;
    }
    pub fn set_shuffle(&mut self, shuffle: bool) {
        self.shuffle = shuffle;
    }
    pub fn set_epochs(&mut self, epochs: usize) {
        self.epochs = epochs;
    }
    pub fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr;
    }
    pub fn fit(&mut self, data: Vec<Vec<D>>, labels: Vec<L>){
        self.data = data;
        self.labels = labels;

        self.weights = vec![vec![0.0; self.data.len()]; 2];  // Two classes: 0 and 1
        self.biases = vec![0.0; 2];

        let binary_labels = self.labels.iter().all(|&x| {
            let val: i64 = x.into();
            val == 0 || val == 1
        });
        panic_labels_not_binary!(!binary_labels, self.model_name);

        if self.shuffle {
            shuffle_data_labels(&mut self.data, &mut self.labels);
        }
        for _ in 0..self.epochs {
            self._fit()
        }
        
    }

    fn _fit(&mut self) {
        for (row, label) in self.data.iter().zip(self.labels.iter()) {
            let features_f64: Vec<f64> = row.iter().map(|x| x.to_f64().unwrap_or(0.0)).collect();
            let pred_class_index = perceptron_predict_idx(&features_f64, &self.weights, &self.biases);

            let actual_class_index: usize = (*label).into() as usize;

            if pred_class_index != actual_class_index {
                update_weights_biases(
                    &mut self.weights,
                    &mut self.biases,
                    self.learning_rate,
                    features_f64,
                    actual_class_index,
                    pred_class_index,
                );
            }
        }
    }

    pub fn predict(&self, sample: Vec<D>) -> L {
        panic_untrained!(self.labels.len() == 0, self.model_name);

        let features_f64: Vec<f64> = sample
            .into_iter()
            .map(|x| x.to_f64().unwrap_or(0.0))
            .collect();

        let class_idx = perceptron_predict_idx(&features_f64, &self.weights, &self.biases);
        
        NumCast::from(class_idx).unwrap()
    }
}

fn perceptron_predict_idx(features: &[f64], weights: &Vec<Vec<f64>>, biases: &Vec<f64>) -> usize {
    let mut best_score = f64::MIN;
    let mut best_idx = 0;

    for (idx, weight_vector) in weights.iter().enumerate() {
        let mut score = biases[idx];
        for (w, &x) in weight_vector.iter().zip(features.iter()) {
            score += w * x;
        }

        if score > best_score {
            best_score = score;
            best_idx = idx;
        }
    }
    best_idx
}

fn update_weights_biases(weights: &mut Vec<Vec<f64>>, biases: &mut Vec<f64>, learning_rate: f64, features_f64: Vec<f64>, actual_class_index: usize, pred_class_index: usize) {
    for (f_idx, &feature) in features_f64.iter().enumerate() {
        weights[actual_class_index][f_idx] += learning_rate * feature;
    }
    biases[actual_class_index] += learning_rate;

    // Update the predicted (incorrect) class
    for (f_idx, &feature) in features_f64.iter().enumerate() {
        weights[pred_class_index][f_idx] -= learning_rate * feature;
    }
    biases[pred_class_index] -= learning_rate;
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper function to construct a default perceptron if needed.
    fn build_perceptron() -> MultiClassPerceptron<f64, i32> {
        MultiClassPerceptron::new()
    }


    #[test]
    fn test_set_penalty() {
        let mut perceptron = build_perceptron();
        let penalty = "l2".to_string();
        perceptron.set_penalty(penalty.clone());
        // Verify penalty was set as expected:
        assert_eq!(perceptron.penalty, penalty);
    }

    #[test]
    fn test_set_alpha() {
        let mut perceptron = build_perceptron();
        let alpha_val = 0.1_f32;
        perceptron.set_alpha(alpha_val);
        assert_eq!(perceptron.alpha, alpha_val);
    }

    #[test]
    fn test_set_epochs() {
        let mut perceptron = build_perceptron();
        let epoch_val = 10_usize;
        perceptron.set_epochs(epoch_val);
        assert_eq!(perceptron.epochs, epoch_val);
    }

    #[test]
    fn test_set_shuffle() {
        let mut perceptron = build_perceptron();
        perceptron.set_shuffle(true);
        assert!(perceptron.shuffle);
    }
    #[test]
    fn test_set_learning_rate() {
        let mut perceptron = build_perceptron();
        let lr = 0.01_f64;
        perceptron.set_learning_rate(lr);
        // Assuming `learning_rate` is accessible to check:
        // If it's private and there's no getter, you could only confirm there's no panic.
        assert_eq!(perceptron.learning_rate, lr);
    }

    #[test]
    fn test_fit() {
        let mut perceptron = build_perceptron();

        // Dummy data and labels (replace with realistic data)
        let data = vec![
            vec![0.0, 1.0],
            vec![1.0, 0.0],
        ];
        let labels = vec![0, 1];

        // Just checking if fit runs without panicking
        perceptron.fit(data, labels);
    }

    #[test]
    #[should_panic]
    fn test_predict_no_fit() {
        let perceptron = build_perceptron();

        let sample = vec![0.5, 0.5];
        // This should panic because the model has not been trained
        let _prediction = perceptron.predict(sample);
        
    }

    #[test]
    fn test_multiclass_prediction() {
        // Build a perceptron with reasonable default parameters
        let mut perceptron = MultiClassPerceptron::<f64, i32>::new();
        perceptron.set_penalty("none".to_string());
        perceptron.set_alpha(0.0);
        perceptron.set_epochs(100);
        perceptron.set_shuffle(false);
        perceptron.set_learning_rate(0.1);

        let training_data = vec![
            // Class 0
            vec![0.0, 0.0],
            vec![0.2, 0.1],
            vec![1.0, 0.5],
            // Class 1
            vec![4.0, 4.5],
            vec![5.0, 5.0],
            vec![5.2, 4.8],
            // Class 2
            vec![9.0, 9.5],
            vec![10.0, 10.0],
            vec![10.5, 9.7],
        ];

        // Corresponding labels (3 classes: 0, 1, 2)
        let training_labels = vec![
            0, 0, 0,
            1, 1, 1,
            2, 2, 2,
        ];

        perceptron.fit(training_data.clone(), training_labels.clone());
        
        for (sample, expected_label) in training_data.iter().zip(training_labels.iter()) {
            let prediction = perceptron.predict(sample.clone());
            assert_eq!(
                prediction, *expected_label,
                "Prediction mismatch for sample {:?}",
                sample
            );
        }

        let test_samples = vec![
            (vec![0.5, 0.0], 0),   // should be near class 0 cluster
            (vec![4.8, 5.1], 1),   // should be near class 1 cluster
            (vec![11.0, 9.8], 2),  // should be near class 2 cluster
        ];
        for (sample, expected) in test_samples {
            let prediction = perceptron.predict(sample.clone());
            assert_eq!(
                prediction, expected,
                "Expected sample {:?} to be in class {}, got {}",
                sample, expected, prediction
            );
        }
    }
    
    #[test]
    fn test_multiclass_prediction_single_vec() {
        let mut perceptron = MultiClassPerceptron::<f64, i32>::new();
        perceptron.set_penalty("none".to_string());
        perceptron.set_alpha(0.0);
        perceptron.set_epochs(100);
        perceptron.set_shuffle(false);
        perceptron.set_learning_rate(0.1);

        let training_data = vec![
            // Class 0
            vec![0.0],
            vec![0.2],
            vec![1.0],
            // Class 1
            vec![4.0],
            vec![5.0],
            vec![5.2],
            // Class 2
            vec![9.0],
            vec![10.0],
            vec![10.5],
        ];

        // Corresponding labels (3 classes: 0, 1, 2)
        let training_labels = vec![
            0, 0, 0,
            1, 1, 1,
            2, 2, 2,
        ];

        perceptron.fit(training_data.clone(), training_labels.clone());

        for (sample, expected_label) in training_data.iter().zip(training_labels.iter()) {
            let prediction = perceptron.predict(sample.clone());
            assert_eq!(
                prediction, *expected_label,
                "Prediction mismatch for sample {:?}",
                sample
            );
        }

        let test_samples = vec![
            (vec![0.5], 0),   // should be near class 0 cluster
            (vec![4.8], 1),   // should be near class 1 cluster
            (vec![11.0], 2),  // should be near class 2 cluster
        ];
        for (sample, expected) in test_samples {
            let prediction = perceptron.predict(sample.clone());
            assert_eq!(
                prediction, expected,
                "Expected sample {:?} to be in class {}, got {}",
                sample, expected, prediction
            );
        }
    }

    #[test]
    fn test_new_binary() {
        let perceptron: BinaryPerceptron<f64, i8> = BinaryPerceptron::new();
        assert_eq!(perceptron.model_name, "BinaryPerceptron");
        assert_eq!(perceptron.penalty, "");
        assert_eq!(perceptron.alpha, 0.0);
        assert!(!perceptron.shuffle);
        assert!(perceptron.data.is_empty());
        assert!(perceptron.labels.is_empty());
        assert!(perceptron.weights.is_empty());
        assert!(perceptron.biases.is_empty());
        assert_eq!(perceptron.learning_rate, 0.3);
        assert_eq!(perceptron.epochs, 10);
    }

    #[test]
    fn test_setters() {
        let mut perceptron: BinaryPerceptron<f64, i8> = BinaryPerceptron::new();

        perceptron.set_penalty("l2".to_string());
        assert_eq!(perceptron.penalty, "l2");

        perceptron.set_alpha(0.5);
        assert_eq!(perceptron.alpha, 0.5);

        perceptron.set_shuffle(true);
        assert!(perceptron.shuffle);

        perceptron.set_epochs(20);
        assert_eq!(perceptron.epochs, 20);

        perceptron.set_learning_rate(0.1);
        assert_eq!(perceptron.learning_rate, 0.1);
    }

    #[test]
    fn test_fit_with_binary_labels() {
        let mut perceptron: BinaryPerceptron<f64, i8> = BinaryPerceptron::new();

        // Create dummy data: two samples with two features each.
        let data = vec![
            vec![0.0, 1.0],
            vec![1.0, 0.0],
        ];
        let labels = vec![0, 1];

        perceptron.fit(data.clone(), labels.clone());

        assert_eq!(perceptron.data, data);
        assert_eq!(perceptron.labels, labels);
    }

    #[test]
    fn test_perceptron_accuracy_linearly_separable() {
        let mut perceptron: BinaryPerceptron<f64, i64> = BinaryPerceptron::new();

        perceptron.set_epochs(100);
        perceptron.set_shuffle(false);
        perceptron.set_learning_rate(0.1);

        // Define the dataset using vector literals.
        // Points where x + y < 0 are labeled 0; x + y ≥ 0 are labeled 1.
        let data = vec![
            vec![-1.0, -1.0],
            vec![-2.0, -1.0],
            vec![-1.5, -0.5],
            vec![-1.0, -2.0],
            vec![ 1.0,  1.0],
            vec![ 2.0,  1.0],
            vec![ 1.5,  0.5],
            vec![ 1.0,  2.0],
        ];
        let labels = vec![0, 0, 0, 0, 1, 1, 1, 1];

        // Fit the perceptron with the dataset.
        perceptron.fit(data, labels.clone());

        assert_eq!(perceptron.predict(vec![1.0, 1.0]), 1);
        assert_eq!(perceptron.predict(vec![-1f64, -1.2]), 0)

    }
}

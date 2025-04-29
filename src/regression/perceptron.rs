//! # Perceptron Regression Algorithm
//!
//! ## Theoretical Background
//!
//! The Perceptron is one of the earliest forms of neural networks, developed in the 1950s. In its regression form:
//!
//! - It models the relationship between inputs and a continuous output through a linear function
//! - It learns incrementally by adjusting weights based on prediction errors
//! - It can be seen as a single-layer neural network with linear activation
//! - It forms the building block for more complex neural network architectures
//!
//! The algorithm works by iteratively adjusting weights and bias based on the error between predicted and actual values, gradually converging toward an optimal model.
//!
//!
//! ## Parameters
//!
//! - `learning_rate`: Step size for weight updates.
//!    - Too small: Slow convergence, may get stuck in local minima
//!    - Too large: May overshoot the optimal solution or diverge
//!    - Typical values: 0.01 to 0.1
//!
//! - `epochs`: Number of complete passes through the training data.
//!    - Too few: Underfitting, insufficient learning
//!    - Too many: Possible overfitting, diminishing returns
//!    - Typical values: 10 to 1000, depending on data complexity
//!
//! - `penalty`: Type of regularization to apply.
//!    - "l1": Lasso regularization (encourages sparsity)
//!    - "l2": Ridge regularization (discourages large weights)
//!    - "none": No regularization
//!
//! - `alpha`: Regularization strength.
//!    - Higher values: Stronger regularization
//!    - Lower values: Weaker regularization
//!    - Typical values: 0.0001 to 0.01
//!
//! - `shuffle`: Whether to shuffle the training data before each epoch.
//!    - true: Helps prevent cycles in convergence, recommended for most cases
//!    - false: Deterministic updates, useful for debugging
//!
//! ## Usage Examples
//!
//! Basic regression with Perceptron:
//!
//! ```rust
//! use rusty_science::regression::RegressionPerceptron;
//!
//! // Create example data
//! let data = vec![
//!     vec![2.0, 3.0], vec![1.0, 5.0], vec![3.0, 2.0],
//!     vec![5.0, 1.0], vec![4.0, 6.0], vec![3.0, 5.0]
//! ];
//! let labels = vec![8.0, 11.0, 7.0, 7.0, 16.0, 13.0];
//!
//! // Create and configure the Perceptron
//! let mut perceptron = RegressionPerceptron::new();
//! perceptron.set_learning_rate(0.01);
//! perceptron.set_epochs(100);
//! perceptron.set_penalty("l2".to_string());
//! perceptron.set_alpha(0.001);
//! perceptron.set_shuffle(true);
//!
//! // Fit the model
//! perceptron.fit(data.clone(), labels);
//!
//! // Make predictions
//! let prediction = perceptron.predict(vec![4.0, 3.0]);
//! println!("Predicted value: {}", prediction);
//! ```
//!
//! ## Performance Characteristics
//!
//! - **Time Complexity**:
//!   - Training: O(n * d * e), where:
//!     - n is the number of training samples
//!     - d is the number of features
//!     - e is the number of epochs
//!   - Prediction: O(d), where d is the number of features
//!
//! - **Space Complexity**: O(d) for storing the weight vector and bias
//!
//! - **Strengths**:
//!   - Simple implementation and intuitive understanding
//!   - Computationally efficient
//!   - Low memory requirements
//!   - Online learning capability (can learn incrementally)
//!   - Works well for linearly separable data
//!   - Serves as a foundation for more complex neural networks
//!
//! - **Weaknesses**:
//!   - Limited to linear relationships between features and target
//!   - Sensitive to feature scaling
//!   - May not converge for non-linearly separable data
//!   - Requires careful tuning of hyperparameters
//!   - Performance can be sensitive to the initial random weight values
//!   - May require feature engineering to capture non-linear relationships
//!

use rand::seq::SliceRandom;

use crate::panic_untrained;
use num::Num;
use num_traits::{NumCast, ToPrimitive};
use rand::thread_rng;

/// A regression perceptron for continuous prediction.
///
/// # Generics
/// * `D` - Data type (must be numeric)
pub struct RegressionPerceptron<D> {
    model_name: String,
    pub(crate) penalty: String,
    pub(crate) alpha: f32,
    pub(crate) shuffle: bool,

    data: Vec<Vec<D>>,
    labels: Vec<f64>,

    weights: Vec<f64>,
    bias: f64,

    learning_rate: f64,
    epochs: usize,
    feature_mins: Vec<f64>,
    feature_maxs: Vec<f64>,

}

impl<D> RegressionPerceptron<D>
where
    D: Num + Copy + Clone + PartialOrd + ToPrimitive + NumCast,
{
    /// Create a new regression perceptron with default settings.
    pub fn new() -> Self {
        Self {
            model_name: "Regression Perceptron".to_string(),
            penalty: "l2".to_string(),
            alpha: 0.0001,
            shuffle: true,
            data: Vec::new(),
            labels: Vec::new(),
            weights: Vec::new(),
            bias: 0.0,
            learning_rate: 0.01,
            epochs: 5,
            feature_mins: vec![],
            feature_maxs: vec![],
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

    /// Fit the regression perceptron with training data and continuous labels.
    ///
    /// * `data` - A vector of samples, where each sample is a vector of features.
    /// * `labels` - A vector of continuous target values.
    pub fn fit(&mut self, mut data: Vec<Vec<D>>, labels: Vec<f64>) {
        if data.is_empty() || labels.is_empty() {
            return;
        }

        let feature_count = data[0].len();
        let mut mins = vec![f64::MAX; feature_count];
        let mut maxs = vec![f64::MIN; feature_count];

        // Find mins and maxs
        for row in &data {
            for (i, val) in row.iter().enumerate() {
                let val_f64 = val.to_f64().unwrap_or(0.0);
                mins[i] = mins[i].min(val_f64);
                maxs[i] = maxs[i].max(val_f64);
            }
        }

        // Normalize the data using min-max scaling
        for row in &mut data {
            for (i, val) in row.iter_mut().enumerate() {
                let val_f64 = val.to_f64().unwrap_or(0.0);
                let denominator = maxs[i] - mins[i];
                if denominator > 0.0 {
                    *val = num_traits::cast((val_f64 - mins[i]) / denominator).unwrap();
                } else {
                    *val = num_traits::cast(0.0).unwrap(); // Avoid division by zero
                }
            }
        }
        self.feature_mins = mins;
        self.feature_maxs = maxs;
        self.weights = vec![0.0; feature_count];
        self.bias = 0.0;

        self.data = data;
        self.labels = labels;
        if self.shuffle {
            let mut rng = thread_rng();
            // Create a vector of indices and shuffle it
            let mut indices: Vec<usize> = (0..self.data.len()).collect();
            indices.shuffle(&mut rng);

            // Reorder data and labels based on shuffled indices
            let mut shuffled_data = Vec::new();
            let mut shuffled_labels = Vec::new();
            for &i in &indices {
                shuffled_data.push(self.data[i].clone());
                shuffled_labels.push(self.labels[i]);
            }
            self.data = shuffled_data;
            self.labels = shuffled_labels;
        }
        for _ in 0..self.epochs {
            self._fit();
        }
    }
    
    fn _fit(&mut self) {
        for (row, &label) in self.data.iter().zip(self.labels.iter()) {
            let features_f64: Vec<f64> = row
                .iter()
                .map(|x| x.to_f64().unwrap_or(0.0))
                .collect();

            let prediction: f64 = self.weights
                .iter()
                .zip(features_f64.iter())
                .map(|(w, x)| w * x)
                .sum::<f64>() + self.bias;

            let error = label - prediction;

            for (i, &x) in features_f64.iter().enumerate() {
                self.weights[i] += self.learning_rate * error * x;
            }
            self.bias += self.learning_rate * error;
        }
        
    }
    
    pub fn score(&self, data: Vec<Vec<D>>, true_labels: Vec<f64>) -> f64 {
        if data.is_empty() || true_labels.is_empty() {
            return f64::NAN;
        }
    
        let preds: Vec<f64> = data.into_iter()
            .map(|sample| self.predict(sample))
            .collect();
    
        let mean_label = true_labels.iter().sum::<f64>() / true_labels.len() as f64;
        let ss_total: f64 = true_labels.iter()
            .map(|y| (y - mean_label).powi(2))
            .sum();
        let ss_res: f64 = preds.iter()
            .zip(true_labels.iter())
            .map(|(y_pred, y_true)| (y_true - y_pred).powi(2))
            .sum();
        
        1.0 - (ss_res / ss_total)
    }

    /// Predict the continuous output for a single sample.
    pub fn predict(&self, sample: Vec<D>) -> f64 {
        panic_untrained!(self.labels.len() == 0, self.model_name);

        let features_f64: Vec<f64> = sample
            .into_iter()
            .enumerate()
            .map(|(i, x)| {
                let val = x.to_f64().unwrap_or(0.0);
                if i < self.feature_mins.len() {
                    let denominator = self.feature_maxs[i] - self.feature_mins[i];
                    if denominator > 0.0 {
                        (val - self.feature_mins[i]) / denominator
                    } else {
                        0.0 // Handle the case where max and min are the same
                    }
                } else {
                    val // fallback if dimensions don't match
                }
            })
            .collect();
        if features_f64.len() != self.weights.len() {
            eprintln!("Warning: Feature dimensions don't match weight dimensions");
            return f64::NAN;
        }
        let prediction = self.weights
            .iter()
            .zip(features_f64.iter())
            .map(|(w, x)| w * x)
            .sum::<f64>() + self.bias;

        if prediction.is_infinite() || prediction.is_nan() {
            eprintln!("Warning: Numerical overflow in prediction");
            return 0.0;
        }
        prediction
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test basic functionality with a simple linear relation: y = 2*x + 1.
    #[test]
    fn test_basic_fit_and_predict() {
        // Training data: x and corresponding y values
        let data = vec![
            vec![1.0_f64],
            vec![2.0_f64],
            vec![3.0_f64],
        ];
        let labels = vec![3.0, 5.0, 7.0]; // 2*x + 1

        let mut model = RegressionPerceptron::<f64>::new();
        // Set hyperparameters
        model.set_learning_rate(0.01);
        model.set_epochs(500);
        model.set_alpha(0.001);
        model.set_penalty("l2".to_string());
        model.set_shuffle(true);

        // Train the model
        model.fit(data, labels);

        // Predict a new sample [4.0] which ideally should be near 9.0.
        let prediction = model.predict(vec![4.0_f64]);
        let expected = 9.0;
        let tolerance = 1.0;
        assert!(
            (prediction - expected).abs() < tolerance,
            "Expected prediction near {}, got {}",
            expected,
            prediction
        );
    }

    // Test with multiple features where the true relationship is:
    // y = 1*x1 + 2*x2 + 3.
    #[test]
    fn test_multiple_features() {
        let data = vec![
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![3.0, 3.0],
        ];
        let labels = vec![
            1.0 * 1.0 + 2.0 * 1.0 + 3.0, // 6.0
            1.0 * 2.0 + 2.0 * 2.0 + 3.0, // 9.0
            1.0 * 3.0 + 2.0 * 3.0 + 3.0, // 12.0
        ];
        let mut model = RegressionPerceptron::<f64>::new();
        model.set_learning_rate(0.01);
        model.set_epochs(500);
        model.fit(data, labels);

        let prediction = model.predict(vec![4.0, 4.0]);
        let expected = 1.0 * 4.0 + 2.0 * 4.0 + 3.0; // 15.0
        let tolerance = 1.0;
        assert!(
            (prediction - expected).abs() < tolerance,
            "Expected prediction near {}, got {}",
            expected,
            prediction
        );
    }

    // Test that the model works with integer data.
    #[test]
    fn test_with_integer_data() {
        // Here we assume a simple relationship: y = x1 + x2 + 1.
        let data = vec![
            vec![1, 2],
            vec![2, 3],
            vec![3, 4],
        ];
        let labels = vec![
            1.0 + 2.0 + 1.0, // 4.0
            2.0 + 3.0 + 1.0, // 6.0
            3.0 + 4.0 + 1.0, // 8.0
        ];
        let mut model = RegressionPerceptron::<i32>::new();
        model.set_learning_rate(0.01);
        model.set_epochs(500);
        model.fit(data, labels);

        let prediction = model.predict(vec![4, 5]);
        let expected = 4.0 + 5.0 + 1.0; // 10.0
        let tolerance = 1.0;
        assert!(
            (prediction - expected).abs() < tolerance,
            "Expected prediction near {}, got {}",
            expected,
            prediction
        );
    }
    
    #[test]
    #[should_panic]
    fn test_predict_before_fit() {
        let model = RegressionPerceptron::<f64>::new();
        model.predict(vec![1.0]);
    }
}

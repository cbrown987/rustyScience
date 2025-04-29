//! Multiple Linear Regression (Gradient Descent)
//!
//! Fits a model: y = w0*x0 + w1*x1 + ... + wn*xn
//!
//! Training logic:
//! - Initialize weights to zero
//! - For each epoch:
//!     - Predict y using dot product of X and weights
//!     - Compute the error (y_pred - y_true)
//!     - Update weights using the gradient of the error
//!       (ie: w_j = w_j - learning_rate * gradient)
//!
//! No matrix inversion used (easier to debug and extend)
//!
//! Supports:
//! - predict(X): returns y values
//! - score(X, y): returns R^2 score
//!
//! Generic over float types like f32 and f64

// mlr.rs

use ndarray::{Array1, Array2};
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

#[derive(Debug, Clone)]
pub struct MultipleLinearRegression<T = f64> { 
    weights: Option<Array1<T>>,
}

impl<T> MultipleLinearRegression<T>
where
    T: Float + FromPrimitive + Debug + 'static,
{
    pub fn new() -> Self {
        Self { weights: None }
    }

    pub fn fit(&mut self, x: &Array2<T>, y: &Array1<T>, n_iter: usize, lr: T) {
        let (n_samples, n_features) = x.dim();
        let mut weights = Array1::<T>::zeros(n_features);

        for _ in 0..n_iter {
            let preds = x.dot(&weights);
            let errors = &preds - y;

            for j in 0..n_features {
                let column = x.column(j);
                let grad = errors.dot(&column) / T::from(n_samples).unwrap();
                weights[j] = weights[j] - lr * grad;
            }
        }

        self.weights = Some(weights);
    }

    pub fn predict(&self, x: &Array2<T>) -> Array1<T> {
        match &self.weights {
            Some(w) => x.dot(w),
            None => panic!("Model must be trained before prediction."),
        }
    }

    pub fn score(&self, x: &Array2<T>, y: &Array1<T>) -> T {
        let preds = self.predict(x);

        if preds.len() != y.len() {
            panic!("Prediction and label lengths do not match.");
        }

        let mean_y = y.mean().expect("Failed to calculate mean of y.");
        let ss_total = y.iter().map(|yi| (*yi - mean_y).powi(2)).fold(T::zero(), |a, b| a + b);
        let ss_res = preds
            .iter()
            .zip(y.iter())
            .map(|(pred, true_y)| (*true_y - *pred).powi(2))
            .fold(T::zero(), |a, b| a + b);

        T::one() - ss_res / ss_total
    }
}

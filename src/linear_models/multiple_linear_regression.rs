use ndarray::{Array1, Array2};
use ndarray::Axis;
use ndarray::stack;
use ndarray_linalg::Solve;

pub struct MultipleLinearRegression {
    coefficients: Array1<f64>,
}

impl MultipleLinearRegression {
    /// Fit the model: coefs = (X^T * X)^-1 * X^T * y
    pub fn fit(x: &Array2<f64>, y: &Array1<f64>) -> Self {
        // Add bias term (column of 1s)
        let ones = Array2::ones((x.nrows(), 1));
        let x_with_bias = stack(Axis(1), &[ones.view(), x.view()])
            .expect("Failed to stack arrays for bias term");

        // Transpose of X with bias
        let xt = x_with_bias.t();

        // X^T * X
        let xtx = xt.dot(&x_with_bias);

        // X^T * y
        let xty = xt.dot(y);

        // Solve for coefficients: (X^T * X)^-1 * X^T * y
        let coefficients = xtx
            .solve(&xty)
            .expect("Matrix is not invertible. Ensure X^T * X is positive definite.");

        // Return the model
        Self { coefficients }
    }

    /// Predict values based on the fitted model
    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        // Add bias term (column of 1s) to X
        let ones = Array2::ones((x.nrows(), 1));
        let x_with_bias = stack(Axis(1), &[ones.view(), x.view()])
            .expect("Failed to stack arrays for bias term");

        // Dot product of X with coefficients to get predictions
        x_with_bias.dot(&self.coefficients)
    }

    /// Calculate R^2 score for the model
    pub fn score(&self, x: &Array2<f64>, y: &Array1<f64>) -> f64 {
        // Mean of y
        let y_mean = y.mean().expect("Failed to calculate mean of y");

        // Predicted values
        let y_pred = self.predict(x);

        // Total sum of squares (SST)
        let ss_total: f64 = y.mapv(|yi| (yi - y_mean).powi(2)).sum();

        // Residual sum of squares (SSR)
        let ss_residual: f64 = y
            .iter()
            .zip(y_pred.iter())
            .map(|(yi, ypi)| (yi - ypi).powi(2))
            .sum();

        // R^2 = 1 - (SSR / SST)
        1.0 - (ss_residual / ss_total)
    }
}
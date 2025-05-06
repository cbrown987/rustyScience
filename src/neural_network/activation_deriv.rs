use crate::neural_network::activation::*;
use ndarray::Array1;
/// Derivative of ReLU activation function
/// ReLU(x) = max(0, x) → ReLU'(x) = 1 if x > 0, else 0
pub fn relu_deriv(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 })
}


//stuff i dont quit undestand
/// Derivative of Sigmoid activation function
/// Sigmoid(x) = 1 / (1 + e^-x)
/// Sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
pub fn sigmoid_deriv(x: &Array1<f64>) -> Array1<f64> {
    let s = sigmoid(x);
    s.mapv(|v| v * (1.0 - v))
}
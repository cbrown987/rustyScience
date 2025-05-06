use ndarray::Array1;

//reul activatio function : max(0, x)
pub fn relu(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|v| v.max(0.0))
}

//deriv of relu function: 1 if x > 0 else 0
pub fn relu_derivative(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 })
}

//sigmoid activation function: 1 / (1 + exp(-x))
pub fn sigmoid(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
}

// deriv of sigmoid function: s(x) * (1 - s(x))
pub fn sigmoid_derivative(x: &Array1<f64>) -> Array1<f64> {
    let s = sigmoid(x);
    s.mapv(|v| v * (1.0 - v))
}
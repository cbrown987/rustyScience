use crate::neural_network::activation::{relu, sigmoid};
use crate::neural_network::activation_deriv::{relu_deriv, sigmoid_deriv};
use crate::neural_network::layer::DenseLayer;
use ndarray::Array1;
use num_traits::ToPrimitive;

pub struct MultiLayerPerceptron<D> {
    pub layers: Vec<DenseLayer<D>>,
}

impl<D> MultiLayerPerceptron<D>
where
    D: ToPrimitive + Copy,
{
    //create a new mlp with changable layer sizes and activation
    pub fn new(layer_sizes: &[usize], use_sigmoid: bool) -> Self {
        let (activation, activation_deriv): (
            fn(&Array1<f64>) -> Array1<f64>,
            fn(&Array1<f64>) -> Array1<f64>,
        ) = if use_sigmoid {
            (sigmoid, sigmoid_deriv)
        } else {
            (relu, relu_deriv)
        };

        let mut layers = Vec::new();
        for i in 0..layer_sizes.len() - 1 {
            layers.push(DenseLayer::<D>::new(
                layer_sizes[i],
                layer_sizes[i + 1],
                activation,
                activation_deriv,
            ));
        }

        Self { layers }
    }

    //forwad pass through all layers 
    pub fn forward(&mut self, input: &[D]) -> Array1<f64> {
        let mut output: Array1<f64> = input.iter().map(|x| x.to_f64().unwrap_or(0.0)).collect();
        for layer in &mut self.layers {
            output = layer.forward(&output);
        }
        output
    }

    //train the network using mse and vanilla gradient descent
    pub fn fit(&mut self, data: &[Vec<D>], labels: &[usize], epochs: usize, learning_rate: f64) {
        let num_classes = self.layers.last().unwrap().biases.len();

        for epoch in 0..epochs {
            let mut total_loss = 0.0;

            for (x, &label) in data.iter().zip(labels.iter()) {
                //convert input to Array1<f64>
                let input: Array1<f64> = x.iter().map(|v| v.to_f64().unwrap_or(0.0)).collect();

                //oh encode label
                let mut target = Array1::zeros(num_classes);
                target[label] = 1.0;

                // fwd pass
                let mut out = input.clone();
                for layer in self.layers.iter_mut() {
                    out = layer.forward(&out);
                }

                //compute MSE loss
                let error = &out - &target;
                total_loss += error.mapv(|e| e * e).sum();

                //backward pass
                let mut delta = error;
                for layer in self.layers.iter_mut().rev() {
                    delta = layer.backward(delta, learning_rate);
                }
            }

            let avg_loss = total_loss / data.len() as f64;
            println!("Epoch {}: loss = {:.4}", epoch + 1, avg_loss);
        }
    }
}

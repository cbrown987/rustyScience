use ndarray::{Array1, Array2, Axis};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use num_traits::ToPrimitive;

pub struct DenseLayer<D> {
    pub weights: Array2<f64>,
    pub biases: Array1<f64>,
    pub input_cache: Array1<f64>,
    pub z_cache: Array1<f64>,
    pub activation: fn(&Array1<f64>) -> Array1<f64>,
    pub activation_deriv: fn(&Array1<f64>) -> Array1<f64>,
    phantom: std::marker::PhantomData<D>,
}

impl<D> DenseLayer<D>
where
    D: ToPrimitive + Copy,
{
    pub fn new(
        input_size: usize,
        output_size: usize,
        activation: fn(&Array1<f64>) -> Array1<f64>,
        activation_deriv: fn(&Array1<f64>) -> Array1<f64>,
    ) -> Self {
        let scale = (1.0 / input_size as f64).sqrt();
        let weights = Array2::random((output_size, input_size), Uniform::new(-scale, scale));
        let biases = Array1::zeros(output_size);

        Self {
            weights,
            biases,
            input_cache: Array1::zeros(input_size),
            z_cache: Array1::zeros(output_size),
            activation,
            activation_deriv,
            phantom: std::marker::PhantomData,
        }
    }

    //fwd pass through layer: z = wx + b  -> a = activation(z)
    pub fn forward(&mut self, input: &Array1<f64>) -> Array1<f64> {
        self.input_cache = input.clone();
        let z = self.weights.dot(input) + &self.biases;
        self.z_cache = z.clone();
        (self.activation)(&z)
    }

    //backward pass using cached forward inputs
    pub fn backward(&mut self, grad_output: Array1<f64>, learning_rate: f64) -> Array1<f64> {
        let dz = grad_output * (self.activation_deriv)(&self.z_cache);
        let dw = dz.view().insert_axis(Axis(1)).dot(&self.input_cache.view().insert_axis(Axis(0)));
        let db = dz.clone();

        self.weights = &self.weights - &(learning_rate * dw);
        self.biases = &self.biases - &(learning_rate * db);
        self.weights.t().dot(&dz)
    }
}

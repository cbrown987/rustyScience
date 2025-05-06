//! # Support Vector Regression (SVR)
//!
//! ## what it does
//! - predicts continuous values
//! - uses margin of tolerance (epsilon)
//! - ignores small errors within margin
//! - penalizes large errors
//! - trains using gradient descent
//!
//! ## hyperparams
//! - `learning_rate`: how fast it learns  
//! - `epochs`: how many passes over data  
//! - `epsilon`: allowed error zone  
//! - `regularization_factor`: controls overfitting  
//!

use std::fmt::Debug;
use num_traits::{Num, NumCast, ToPrimitive};

pub struct SVR<D>
where
    D: Num + Copy + Clone + PartialOrd + ToPrimitive + Debug,
{
    data: Vec<Vec<D>>,
    targets: Vec<f64>,
    weights: Vec<f64>,
    bias: f64,
    epochs: usize,
    learning_rate: f64,
    epsilon: f64,
    regularization_factor: f64,
}

impl<D> SVR<D>
where
    D: Num + Copy + Clone + PartialOrd + ToPrimitive + Debug,
{
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            targets: Vec::new(),
            weights: Vec::new(),
            bias: 0.0,
            epochs: 1000,
            learning_rate: 0.001,
            epsilon: 0.1,
            regularization_factor: 0.01,
        }
    }

    // set training params
    pub fn set_epochs(&mut self, epochs: usize) {
        self.epochs = epochs;
    }

    pub fn set_learning_rate(&mut self, rate: f64) {
        self.learning_rate = rate;
    }

    pub fn set_epsilon(&mut self, epsilon: f64) {
        self.epsilon = epsilon;
    }

    pub fn set_regularization_factor(&mut self, reg: f64) {
        self.regularization_factor = reg;
    }

    // train model
    pub fn fit(&mut self, data: Vec<Vec<D>>, targets: Vec<f64>) {
        self.data = data;
        self.targets = targets;

        let n_samples = self.data.len();
        let n_features = self.data[0].len();
        self.weights = vec![0.0; n_features];
        self.bias = 0.0;

        let eta = self.learning_rate;
        let c = self.regularization_factor;
        let eps = self.epsilon;

        for _ in 0..self.epochs {
            for i in 0..n_samples {
                let xi = &self.data[i];
                let yi = self.targets[i];

                // prediction
                let mut pred = self.bias;
                for j in 0..n_features {
                    pred += self.weights[j] * xi[j].to_f64().unwrap();
                }

                let error = pred - yi;

                // outside epsilon tube
                if error.abs() > eps {
                    let sign = if error > 0.0 { 1.0 } else { -1.0 };
                    for j in 0..n_features {
                        self.weights[j] -= eta * (c * sign * xi[j].to_f64().unwrap() + self.weights[j]);
                    }
                    self.bias -= eta * c * sign;
                } else {
                    // inside tube - apply decay
                    for j in 0..n_features {
                        self.weights[j] -= eta * self.weights[j];
                    }
                }
            }
        }
    }

    // predict single point
    pub fn predict(&self, input: Vec<D>) -> f64 {
        let mut pred = self.bias;
        for (wi, xi) in self.weights.iter().zip(input.iter()) {
            pred += wi * xi.to_f64().unwrap();
        }
        pred
    }

    // r2 score
    pub fn score(&self, test_x: Vec<Vec<D>>, test_y: Vec<f64>) -> f64 {
        let mean_y: f64 = test_y.iter().sum::<f64>() / test_y.len() as f64;
        let mut ss_tot = 0.0;
        let mut ss_res = 0.0;

        for (xi, yi) in test_x.iter().zip(test_y.iter()) {
            let pred = self.predict(xi.clone());
            ss_tot += (yi - mean_y).powi(2);
            ss_res += (yi - pred).powi(2);
        }

        1.0 - ss_res / ss_tot
    }

    // get weights
    pub fn weights(&self) -> &Vec<f64> {
        &self.weights
    }

    // get bias
    pub fn bias(&self) -> f64 {
        self.bias
    }
}
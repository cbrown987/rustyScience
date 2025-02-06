use ndarray::{Array1, Array2, Axis};
use crate::utils::inverse_matrix;

pub struct MultipleLinearRegression {
    coefficients: Option<Array2<f64>>,
}

impl MultipleLinearRegression {
    pub fn new() -> Self {
        MultipleLinearRegression { coefficients: None }
    }

    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) {
        let xt = x.t().to_owned();
        let xt_x = xt.dot(x);
        let xt_y = xt.dot(&y.insert_axis(Axis(1)));

        if let Some(xt_x_inv) = inverse_matrix(&xt_x) {
            let theta = xt_x_inv.dot(&xt_y);
            self.coefficients = Some(theta);
        } else {
            panic!("Matrix inversion failed.");
        }
    }

    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        if let Some(ref theta) = self.coefficients {
            x.dot(theta).column(0).to_owned()
        } else {
            panic!("Model is not trained. Call `fit` first.");
        }
    }

    pub fn get_coefficients(&self) -> Option<&Array2<f64>> {
        self.coefficients.as_ref()
    }
}
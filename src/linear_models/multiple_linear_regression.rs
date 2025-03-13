use ndarray::{Array1, Array2, Axis};
use crate::common::utils::inverse_matrix;
use crate::{panic_matrix_inversion, panic_dimension_mismatch, panic_missing_coefficients};


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
        let xt_y = y.clone().insert_axis(Axis(1));

        panic_dimension_mismatch!(xt_x.nrows() != xt_x.ncols(), "square matrix", format!("{:?}x{:?}", xt_x.nrows(), xt_x.ncols()));


        if let Some(xt_x_inv) = inverse_matrix(&xt_x) {
            let theta = 
            xt_x_inv.dot(&xt_y);
            self.coefficients = Some(theta);
        } else {
            panic_matrix_inversion!(true, "X^T * X");
        }
    }

    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        if let Some(ref theta) = self.coefficients {
            x.dot(theta).column(0).to_owned()
        } else {
            panic_missing_coefficients!();
        }
    }

    pub fn get_coefficients(&self) -> Option<&Array2<f64>> {
        self.coefficients.as_ref()
    }
}
//use crate::common::utils;
//use crate::common::test_utils;

pub struct SimpleLinearRegression {
    pub slope: f64,
    pub intercept: f64,
}


impl SimpleLinearRegression {
    pub fn new() -> Self {
        SimpleLinearRegression {
            slope: 0.0,
            intercept: 0.0,
        }
    }
    //least squares

    pub fn fit(&mut self, x: &[f64], y: &[f64]) {
        let n = x.len() as f64;
        let sum_x = x.iter().sum::<f64>();
        let sum_y = y.iter().sum::<f64>();
        let xy_pairs = x.iter().zip(y.iter()); //pair the two elements up
        let xy_products: Vec<f64> = xy_pairs.map(|(&xi, &yi)| xi * yi).collect();   //Multiply the pairs together and save in vec
        let sum_xy: f64 = xy_products.iter().sum();
        let sum_x_squared = x.iter().map(|xi| xi * xi).sum::<f64>();

        //slope + intercept
        self.slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x * sum_x);
        self.intercept = (sum_y - self.slope * sum_x) / n;
    }
        //eval method
    pub fn predict(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&xi| self.slope * xi + self.intercept).collect()
    }
    // cooper you may need to explain r^2 method but I made this one
    pub fn score(&self, x: &[f64], y: &[f64]) -> Vec<f64> {        
        let y_mean = y.iter().sum::<f64>() / y.len() as f64;
        let ss_total = y.iter().map(|yi| (yi - y_mean).powi(2)).sum::<f64>();
        let y_pred = self.predict(x);
        let ss_residual = y.iter().zip(y_pred.iter()).map(|(yi, ypi)| (yi - ypi).powi(2)).sum::<f64>();

        let r_squared: f64 = 1.0 - (ss_residual / ss_total);//r^2 val
        vec![r_squared]
    }
}
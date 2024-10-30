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

    //***REMOVE PREFIX UNDERSCORES BEFORE IMPLEMENTATION***
    pub fn fit(&mut self, x: &[f64], y: &[f64]) {
        let _n = x.len() as f64;
        let _sum_x = x.iter().sum::<f64>();
        let _sum_y = y.iter().sum::<f64>();
        let xy_pairs = x.iter().zip(y.iter()); //pair the two elements up
        let xy_products: Vec<f64> = xy_pairs.map(|(&xi, &yi)| xi * yi).collect();   //Multiply the pairs together and save in vec
        let _sum_xy: f64 = xy_products.iter().sum();

        //slope + intercept

        //eval method

        //prediction method
    }
}


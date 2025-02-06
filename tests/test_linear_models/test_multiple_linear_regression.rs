use rusty_science::linear_models::multiple_linear_regression::MultipleLinearRegression;
use ndarray::array;
use ndarray::{Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;


fn test_multiple_linear_regression() {
    let mut rng = rand::thread_rng();
    let num_samples = 100;

    //dataset
    let x1: Array1<f64> = Array1::random_using(num_samples, Uniform::new(0.0, 10.0), &mut rng);
    let x2: Array1<f64> = Array1::random_using(num_samples, Uniform::new(0.0, 20.0), &mut rng);
    let noise: Array1<f64> = Array1::random_using(num_samples, Uniform::new(-2.0, 2.0), &mut rng);

    let y: Array1<f64> = &x1 * 3.0 + &x2 * 2.0 + noise;

    let ones = Array2::ones((num_samples, 1));
    let x1_col = x1.clone().insert_axis(Axis(1));
    let x2_col = x2.clone().insert_axis(Axis(1));
    let x = ndarray::concatenate(Axis(1), &[ones.view(), x1_col.view(), x2_col.view()])
        .expect("Concatenation failed");

    //test mlr
    let mut mlr = MultipleLinearRegression::new();
    mlr.fit(&x, &y);
    let y_pred_multi = mlr.predict(&x);

    let mse_multi = (&y - &y_pred_multi).mapv(|v| v.powi(2)).mean().unwrap();

    println!("\nMultiple Linear Regression:");
    println!("Coefficients: {:?}", mlr.get_coefficients());
    println!("MSE: {}", mse_multi);
}
use ndarray::array;
use rusty_science::linear_models::multiple_linear_regression::MultipleLinearRegression;
use rusty_science::linear_models::simple_linear_regression::SimpleLinearRegression;

// ########## MLR DEMO ############

fn run_mlr_demo() {
    println!("Multiple Linear Regression Demo");

    let x_mlr = array![
        [1.0, 2.0],
        [2.0, 0.0],
        [3.0, 1.0],
        [0.0, 2.0]
    ];
    let y_mlr = array![13.0, 9.0, 14.0, 11.0];
    let mut mlr_model = MultipleLinearRegression::new();
    mlr_model.fit(&x_mlr, &y_mlr, 1000, 0.01_f64);

    let preds_mlr = mlr_model.predict(&x_mlr);
    println!("MLR Predictions: {:?}", preds_mlr);

    let r2_mlr = mlr_model.score(&x_mlr, &y_mlr);
    println!("R2: {:.4}\n", r2_mlr);
}

// ######SLR Demo##########

fn run_slr_demo() {
    println!("Simple Linear Regression (SLR) Demo");

    let x_slr = array![1.0, 2.0, 3.0, 4.0];
    let y_slr = array![2.0, 4.1, 6.0, 8.1];

    let mut slr_model = SimpleLinearRegression::new();
    slr_model.fit(&x_slr, &y_slr);

    let preds_slr = slr_model.predict(&x_slr);
    println!("SLR Predictions: {:?}", preds_slr);

    let r2_slr = slr_model.score(&x_slr, &y_slr).expect("Failed to score SLR");
    println!("SLR R2: {:.4}", r2_slr);

    println!("SLR Slope: {:.4}", slr_model.slope());
    println!("SLR Intercept: {:.4}\n", slr_model.intercept());
}



fn main() {
    run_mlr_demo();
    run_slr_demo();
}
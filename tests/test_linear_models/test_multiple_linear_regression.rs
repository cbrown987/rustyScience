use rusty_science::linear_models::multiple_linear_regression::MultipleLinearRegression;
use ndarray::array;

fn test_multiple_linear_regression() {
    let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
    let y = array![3.0, 5.0, 7.0, 9.0];

    let model = MultipleLinearRegression::fit(&x, &y);
    let x_new = array![[5.0, 6.0], [6.0, 7.0]];
    let predictions = model.predict(&x_new);

    assert!(predictions.abs_diff_eq(&array![11.0, 13.0], 1e-6));

    let r2 = model.score(&x, &y);
    assert!((r2 - 1.0).abs() < 1e-6);
}
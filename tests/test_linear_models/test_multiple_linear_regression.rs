use rusty_science::linear_models::multiple_linear_regression::MultipleLinearRegression;
use ndarray::{array, Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;

#[test]
fn test_fit_predict_and_score() {
    let mut rng = rand::thread_rng();
    let n = 100;

    let x1 = Array1::random_using(n, Uniform::new(0.0, 10.0), &mut rng);
    let x2 = Array1::random_using(n, Uniform::new(0.0, 10.0), &mut rng);
    let noise = Array1::random_using(n, Uniform::new(-1.0, 1.0), &mut rng);
    let y = &x1 * 2.0 + &x2 * 3.0 + &noise;

    let ones = Array2::ones((n, 1));
    let x = ndarray::stack(
        Axis(1),
        &[
            ones.view(),
            x1.clone().insert_axis(Axis(1)).view(),
            x2.clone().insert_axis(Axis(1)).view(),
        ],
    )
    .unwrap();

    let mut model = MultipleLinearRegression::new();
    model.fit(&x, &y);
    let preds = model.predict(&x);

    let mse = (&y - &preds).mapv(|v| v.powi(2)).mean().unwrap();
    assert!(mse < 5.0);

    let coeffs = model.get_coefficients().unwrap();
    assert_eq!(coeffs.nrows(), 3); // bias + 2 weights
}

#[test]
fn test_prediction_length_matches_input() {
    let x = array![[1.0, 2.0], [3.0, 4.0]];
    let y = array![8.0, 18.0];

    let mut model = MultipleLinearRegression::new();
    model.fit(&x, &y);
    let preds = model.predict(&x);
    assert_eq!(preds.len(), y.len());
}

#[test]
#[should_panic(expected = "Model is not trained. Call `fit` first.")]
fn test_predict_before_fit_panics() {
    let x = array![[1.0, 2.0], [3.0, 4.0]];
    let model = MultipleLinearRegression::new();
    let _ = model.predict(&x);
}
use rusty_science::regression::SVR;
use rusty_science::common::test_utils::load_iris;
// test basic training and prediction
#[test]
fn test_fit_and_predict() {
    let data = vec![vec![1.0], vec![2.0], vec![3.0]];
    let targets = vec![3.0, 5.0, 7.0]; // y = 2x + 1

    let mut model = SVR::new();
    model.set_epochs(500);
    model.fit(data.clone(), targets.clone());

    let pred = model.predict(vec![4.0]);
    assert!((pred - 9.0).abs() < 0.5); // expect around 9
}

// test r2 score is high for linear data
#[test]
fn test_score_on_linear_data() {
    let data = vec![vec![1.0], vec![2.0], vec![3.0]];
    let targets = vec![3.0, 5.0, 7.0];

    let mut model = SVR::new();
    model.set_epochs(500);
    model.fit(data.clone(), targets.clone());

    let score = model.score(data, targets);
    assert!(score > 0.9); // should be high
}

// test with no training
#[test]
fn test_no_training() {
    let data = vec![vec![1.0], vec![2.0]];
    let targets = vec![3.0, 5.0];

    let mut model = SVR::new();
    model.set_epochs(0); // no training
    model.fit(data, targets);

    let pred = model.predict(vec![2.0]);
    assert!(pred.abs() < 0.01); // should stay near 0
}

// test on single point
#[test]
fn test_single_point_fit() {
    let data = vec![vec![2.0]];
    let targets = vec![7.0];

    let mut model = SVR::new();
    model.set_epochs(500);
    model.fit(data, targets);

    let pred = model.predict(vec![2.0]);
    assert!((pred - 7.0).abs() < 0.2);
}

fn test_svr_on_iris() {
    let iris = load_iris();
    let (data, _labels) = iris.to_numerical_labels();

    // use first 4 cols as input (sepal+petal length/width)
    let x: Vec<Vec<f64>> = data.iter().map(|row| row[..4].to_vec()).collect();

    // use petal width (4th col) as regression target
    let y: Vec<f64> = data.iter().map(|row| row[3]).collect();

    let mut model = SVR::new();
    model.set_epochs(1000);
    model.set_learning_rate(0.01);
    model.set_epsilon(0.1);
    model.set_regularization_factor(0.01);

    model.fit(x.clone(), y.clone());

    let score = model.score(x, y);
    assert!(score > 0.8); // decent fit
}
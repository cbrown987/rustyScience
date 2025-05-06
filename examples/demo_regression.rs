use rusty_science::regression::SVR;
use rusty_science::regression::knn::KNNRegression;
use rusty_science::regression::perceptron::RegressionPerceptron;
//litterally have no idea whjy this wont work. 

use rusty_science::data::datasets::{load_iris, load_housing};


pub fn run_svr_demo() {
    let iris = load_iris();
    let (data, _labels) = iris.to_numerical_labels();

    let x: Vec<Vec<f64>> = data.iter().map(|row| row[..4].to_vec()).collect();
    let y: Vec<f64> = data.iter().map(|row| row[3]).collect(); // Petal width

    let mut model = SVR::new();
    model.set_epochs(1000);
    model.set_learning_rate(0.01);
    model.set_epsilon(0.1);
    model.set_regularization_factor(0.01);
    model.fit(x.clone(), y.clone());
    let score = model.score(x, y);

    println!("R2 Score on Iris dataset: {:.4}", score);
}

pub fn run_perceptron_demo() {
    println!("Running Regression Perceptron Demo...");

    let (data, labels) = load_housing().to_numerical_values();
    let mut model = RegressionPerceptron::new();
    model.set_epochs(100);
    model.set_learning_rate(0.001);
    model.fit(data.clone(), labels.clone());
    let r2 = model.score(data, labels);

    println!("Regression Perceptron R² Score on Housing dataset: {:.4}\n", r2);
}

pub fn run_knn_regression_demo() {
    println!("Running KNN Regression Demo...");

    let (data, labels) = load_housing().to_numerical_values();
    let mut model = KNNRegression::new(5); // 5 nneigbnors
    model.fit(data.clone(), labels.clone());

    let r2 = model.score(data, labels);

    println!("KNN Regression R2 Score on Housing dataset: {:.4}\n", r2);
}


fn main() {
    println!("Running Regression Models Demo...\n");

    run_perceptron_demo();
    run_svr_demo();
    run_knn_demo();
    run_decision_tree_demo();
}
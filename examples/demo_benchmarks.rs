// use rusty_science::benches::regression::*;
// use rusty_science::benches::classification::*;
// use std::time::Instant;

// pub fn run() {

// //#######REGRESSION########//
//     println!("Regression Benchmarks:");

//     let start = Instant::now();
//     simple_linear_regression_bench();
//     println!("Simple Linear Regression... Done in {:.3?}", start.elapsed());

//     let start = Instant::now();
//     multiple_linear_regression_bench();
//     println!("Multiple Linear Regression... Done in {:.3?}", start.elapsed());

//     let start = Instant::now();
//     svr_bench();
//     println!("SVR... Done in {:.3?}", start.elapsed());

// //#######CLASSIFICATION########//


//     println!("\nClassification Benchmarks:");

//     let start = Instant::now();
//     perceptron_bench();
//     println!("Perceptron... Done in {:.3?}", start.elapsed());

//     let start = Instant::now();
//     svc_bench();
//     println!("SVC... Done in {:.3?}", start.elapsed());

//     let start = Instant::now();
//     knn_bench();
//     println!("KNN Classifier... Done in {:.3?}", start.elapsed());

//     let start = Instant::now();
//     decision_tree_bench();
//     println!("Decision Tree... Done in {:.3?}", start.elapsed());
// }
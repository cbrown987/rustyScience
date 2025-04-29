use rusty_science::classification::perceptron::BinaryPerceptron;
use rusty_science::classification::svc::BinarySVC;
use rusty_science::classification::knn::KNNClassifier;
use rusty_science::classification::tree::TreeClassifier;

use rusty_science::data::datasets::{load_iris, load_brest_cancer};

use rusty_science::metrics::accuracy_score;

// === Perceptron Demo ===
fn run_perceptron_demo() {
    println!("=== Binary Perceptron Demo (Breast Cancer Dataset) ===");

    let breast_cancer_data = load_brest_cancer();
    let (mut data, mut labels) = breast_cancer_data.to_numerical_labels();

    let target_feature = data.remove(0);
    let target_label = labels.remove(0);

    let mut perceptron = BinaryPerceptron::new();
    perceptron.set_epochs(1000);
    perceptron.fit(data.clone(), labels.clone());

    let prediction = perceptron.predict(target_feature.clone());

    println!("Expected Label: {}", target_label);
    println!("Predicted Label: {}\n", prediction);
}

// === SVC Demo ===
fn run_svc_demo() {
    println!("=== Binary Support Vector Classifier Demo (Iris Dataset) ===");

    let iris_data = load_iris();
    let (data, labels): (Vec<Vec<f64>>, Vec<i64>) = iris_data.to_numerical_labels();




    let binary_labels: Vec<i64> = labels
        .into_iter()
        .map(|label| if label == 0 { 0 } else { 1 })
        .collect();


    let mut svc = BinarySVC::<f64, i64>::new();
    svc.set_epochs(1000);

    svc.fit(data.clone(), binary_labels.clone());


    let target = vec![1.5, 1.5, 1.5, 1.5];
    let prediction = svc.predict(target.clone());

    println!("Predicted Label for {:?}: {}\n", target, prediction);
}

// === KNN Classifier Demo ===
fn run_knn_demo() {
    println!("=== K-Nearest Neighbors Classifier Demo (Iris Dataset) ===");

    let iris_data = load_iris();
    let (mut data, labels) = iris_data.to_numerical_labels();



    data = data.into_iter()
        .map(|vec| vec![vec[0], vec[1]])
        .collect();

    let mut knn = KNNClassifier::new(3);

    knn.fit(data.clone(), labels.clone());


    let target = vec![1.5, 1.5];

    let prediction = knn.predict(target.clone());

    println!("Predicted Label for {:?}: {}\n", target, prediction);
}

// === Decision Tree Demo ===
fn run_decision_tree_demo() {
    println!("=== Decision Tree Demo (Iris Dataset) ===");

    let iris_data = load_iris();
    let (data, labels) = iris_data.to_numerical_labels();

    let mut tree = TreeClassifier::new();
    tree.fit(data.clone(), labels.clone());

    let target = vec![1.5, 1.5, 1.5, 1.5];
    let prediction = tree.predict(target.clone());

    println!("Predicted Label for {:?}: {}\n", target, prediction);
}

// === Main ===
fn main() {
    println!("Running Classification Models Demo...\n");

    run_perceptron_demo();
    run_svc_demo();
    run_knn_demo();
    run_decision_tree_demo();
}
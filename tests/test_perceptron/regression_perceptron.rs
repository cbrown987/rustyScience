use rusty_science::regression::RegressionPerceptron;
use rusty_science::data::load_housing;
#[test]
pub(crate) fn test_perceptron_regression_integration() {
    let (mut data, mut labels) = load_housing().to_numerical_values();
    

    let target = data.remove(0);
    let target_label = labels.remove(0);

    let mut perceptron = RegressionPerceptron::new();
    perceptron.set_epochs(100); 
    perceptron.set_learning_rate(0.001); 
    perceptron.fit(data, labels);
    let prediction = perceptron.predict(target);

    let relative_error = (prediction - target_label).abs() / target_label.abs();
    assert!(relative_error < 0.5,
            "Prediction: {}, Actual: {}, Relative Error: {}",
            prediction, target_label, relative_error);
}

use rusty_science::classification::perceptron::MultiClassPerceptron;
use rusty_science::data::datasets::load_iris;

#[test]
fn test_multi_class_perceptron() {
    let iris_data = load_iris();
    let (data, labels) = iris_data.to_numerical_labels();
    let target = vec![1.5, 1.5];
    
    let mut perceptron: MultiClassPerceptron<f64, i64> = MultiClassPerceptron::new();
    perceptron.set_epochs(100);
    
    perceptron.fit(data, labels);
    let prediction = perceptron.predict(target);
    
    assert_eq!(prediction, 0)
}
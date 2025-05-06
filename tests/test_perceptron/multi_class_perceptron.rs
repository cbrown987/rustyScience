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


// test with more training
#[test]
fn test_multi_class_perceptron_more_epochs() {
    let iris_data = load_iris();
    let (mut data, mut labels) = iris_data.to_numerical_labels();

    let target_feature = data.remove(0);
    let target_label = labels.remove(0);

    let mut perceptron: MultiClassPerceptron<f64, i64> = MultiClassPerceptron::new();
    perceptron.set_epochs(1000); // more training

    perceptron.fit(data, labels);
    let prediction = perceptron.predict(target_feature);

    assert_eq!(prediction, target_label);
}

// test prediction is always in 0, 1, or 2
#[test]
fn test_multi_class_prediction_range() {
    let iris_data = load_iris();
    let (data, labels) = iris_data.to_numerical_labels();

    let mut perceptron: MultiClassPerceptron<f64, i64> = MultiClassPerceptron::new();
    perceptron.set_epochs(100);
    perceptron.fit(data.clone(), labels.clone());

    for x in data.iter().take(10) {
        let pred = perceptron.predict(x.clone());
        assert!(pred == 0 || pred == 1 || pred == 2);
    }
}
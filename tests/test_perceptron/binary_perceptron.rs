use rusty_science::classification::perceptron::BinaryPerceptron;
use rusty_science::data::datasets::load_brest_cancer;

#[test]
fn test_multi_class_perceptron() {
    let brest_cancer_data = load_brest_cancer();
    let (mut data, mut labels) = brest_cancer_data.to_numerical_labels();
    let target_feature = data.remove(0);
    let target_label = labels.remove(0);
    let mut perceptron: BinaryPerceptron<f32, i32> = BinaryPerceptron::new();
    perceptron.set_epochs(100);

    perceptron.fit(data, labels);
    let prediction = perceptron.predict(target_feature);

    assert_eq!(prediction, target_label)
}

#[test]
fn test_binary_perceptron_second_row() {
    let brest_cancer_data = load_brest_cancer();
    let (mut data, mut labels) = brest_cancer_data.to_numerical_labels();
    let target_feature = data.remove(1); 
    let target_label = labels.remove(1);
    let mut model = BinaryPerceptron::new();
    model.set_epochs(100);
    model.fit(data, labels);

    let prediction = model.predict(target_feature);
    assert_eq!(prediction, target_label);
}

// test with longer training
#[test]
fn test_binary_perceptron_more_epochs() {
    let brest_cancer_data = load_brest_cancer();
    let (mut data, mut labels) = brest_cancer_data.to_numerical_labels();
    let target_feature = data.remove(0);
    let target_label = labels.remove(0);
    let mut model = BinaryPerceptron::new();
    model.set_epochs(1000); // longer training
    model.fit(data, labels);

    let prediction = model.predict(target_feature);
    assert_eq!(prediction, target_label);
}

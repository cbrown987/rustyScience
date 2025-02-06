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
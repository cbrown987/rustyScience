use rusty_science::classification::TreeClassifier;
use rusty_science::data::datasets::load_iris;

#[test]
pub(crate) fn test_tree_classification_integration() {
    let iris_data = load_iris();
    let (data, labels) = iris_data.to_numerical_labels();

    let target = vec![1.5, 1.5, 1.5, 1.5];

    let mut tree = TreeClassifier::new();
    tree.fit(data, labels);
    let prediction = tree.predict(target);

    assert_eq!(prediction, 0);
}

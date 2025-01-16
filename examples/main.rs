use rusty_science::classification::TreeClassifier;
use rusty_science::data::generate_labeled_data;

fn main() {

    let (data, labels) = generate_labeled_data::<f64, f64>((1000, 1000));
    let data_cloned = data.clone();
    let target_label = labels[0].clone();
    let mut tree = TreeClassifier::new();
    tree.fit(data, labels);
    let prediction = tree.predict(data_cloned[0].clone());
    println!("{:?}", prediction);
    println!("{:?}", target_label);
}
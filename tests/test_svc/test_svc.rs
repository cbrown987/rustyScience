use rusty_science::classification::BinarySVC;
use rusty_science::data::load_iris;

#[test]
fn test_binary_svc() {
    let iris_data = load_iris();
    let (data, labels): (Vec<Vec<f64>>, Vec<i64>) = iris_data.to_numerical_labels();
    
    // Convert iris data to binary problem by combining 2 labels 
    let binary_labels: Vec<i64> = labels
        .into_iter()
        .map(|label| if label == 0 { 0 } else { 1 }) // Map label 0 to 0, others to 1
        .collect();

    let binary_data = data; // No change to data

    let target = vec![1.5, 1.5, 1.5, 1.5];
    
    let mut svc = BinarySVC::<f64, i64>::new();
    svc.set_epochs(1000);
    svc.fit(binary_data, binary_labels);
    let prediction = svc.predict(target);
    
    assert_eq!(prediction, 0);
}
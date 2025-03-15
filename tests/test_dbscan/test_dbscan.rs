use rusty_science::clustering::DBSCAN;
use rusty_science::data::load_iris;

#[test]
fn test_dbscan_predict() {
    let iris_data = load_iris();
    let (data, labels) = iris_data.to_numerical_labels();
    
    let target = 
        vec![
            vec![1.5, 1.5],
            vec![2.0, 3.1],
            vec![3f64 , 3.1]
        ];
    
    let eps = 0.5;
    let min_samples = 3;
    
    let mut dbscan = DBSCAN::new();
    dbscan.set_eps(eps);
    dbscan.set_min_samples(min_samples);
    dbscan.fit(data, labels);
    
    let prediction = dbscan.predict(target);
    if prediction.len() > 0 {
        assert_eq!(prediction[0], 0);
    }
    else {
        assert!(false);
    }
}

#[test]
fn test_dbscan_predict_one() {
    let iris_data = load_iris();
    let (data, labels) = iris_data.to_numerical_labels();
    
    let target = vec![2.1, 3.1];
    
    let eps = 0.5;
    let min_samples = 3;
    let mut dbscan = DBSCAN::new();
    dbscan.set_eps(eps);
    dbscan.set_min_samples(min_samples);
    dbscan.fit(data, labels);
    
    let prediction = dbscan.predict_one(target);
    assert_eq!(prediction, 0);
}

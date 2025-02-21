use rusty_science::classification::knn::KNNClassifier;

// TEST FUNC

#[test]
fn test_fit_and_predict() {
    let mut model = SimpleLinearRegression::new();
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    
    model.fit(&x, &y);
    let predictions = model.predict(&x);
    

    for (yi, ypi) in y.iter().zip(predictions.iter()) {
        assert!((yi - ypi).abs() < 1e-6);  // allows fo rsmall error
    }
}

#[test]
fn test_score() {
    let mut model = SimpleLinearRegression::new();
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    
    model.fit(&x, &y);
    let r_squared = model.score(&x, &y);
    
    assert!((r_squared - 1.0).abs() < 1e-6);
}
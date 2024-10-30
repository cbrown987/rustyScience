use rustyScience::metrics::accuracy_score::{accuracy_score, accuracy_score_normalize};

#[test]
fn test_accuracy_score() {
    let data_test_one = vec![1,2,3,4,5,6,7];
    let data_pred_one = vec![1,2,3,4,5,6,7];
    let accuracy = accuracy_score(data_test_one, data_pred_one);
    assert_eq!(accuracy, 7);
}

#[test]
fn test_accuracy_score_normalized() {
    let data_test_three = vec![1,2,3,4,5,6,7,8,9,10];
    let data_pred_three = vec![1,2,3,4,5,6,7];
    let accuracy = accuracy_score_normalize(data_test_three, data_pred_three);
    assert_eq!(accuracy, 0.7);
}
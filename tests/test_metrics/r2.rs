use rand::{thread_rng, Rng};
use rustyScience::metrics::r2::r2;
#[test]
fn test_random_data_with_high_correlation() {
    let mut rng = thread_rng();
    let y_true: Vec<f64> = (0..100).map(|x| x as f64).collect();
    let y_pred: Vec<f64> = y_true.iter().map(|&y| y + rng.gen_range(-0.5..0.5)).collect();

    let r2 = r2(y_true, y_pred);
    assert!(r2 > 0.95, "Expected R^2 > 0.95, got {}", r2);
}

#[test]
fn test_random_data_with_low_correlation() {
    let mut rng = thread_rng();
    let y_true: Vec<f64> = (0..100).map(|x| x as f64).collect();
    let y_pred: Vec<f64> = (0..100).map(|_| rng.gen_range(0.0..100.0)).collect();

    let r2 = r2(y_true, y_pred);
    assert!(r2 < 0.1, "Expected R^2 < 0.1, got {}", r2);
}

#[test]
fn test_large_dataset() {
    let y_true: Vec<f64> = (0..10_000).map(|x| x as f64 * 2.0).collect();
    let y_pred: Vec<f64> = y_true.iter().map(|&y| y + 1.0).collect();

    let r2 = r2(y_true, y_pred);
    assert!(r2 > 0.99, "Expected R^2 > 0.99, got {}", r2);
}

#[test]
fn test_perfect_linear_relationship() {
    let y_true: Vec<f64> = (0..100).map(|x| x as f64).collect();
    let y_pred: Vec<f64> = y_true.clone();

    let r2 = r2(y_true, y_pred);
    assert!((r2 - 1.0).abs() < 1e-9, "Expected R^2 close to 1 for perfect linear relationship, got {}", r2);
}

#[test]
fn test_nonlinear_relationship() {
    let y_true: Vec<f64> = (0..100).map(|x| x as f64).collect();
    let y_pred: Vec<f64> = y_true.iter().map(|&y| y.powi(2) * 0.01).collect();

    let r2 = r2(y_true, y_pred);
    assert!(r2 < 1.0, "Expected R^2 < 1 for nonlinear relationship, got {}", r2);
}

#[test]
pub(crate) fn test_all_r2(){
    test_random_data_with_high_correlation();
    test_random_data_with_low_correlation();
    test_large_dataset();
    test_perfect_linear_relationship();
    test_nonlinear_relationship();
}


use rusty_science::linear_models::multiple_linear_regression::SimpleLinearRegression;


#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_fit_and_predict() {
        let x = array![1.0, 2.0, 3.0];
        let y = array![2.0, 4.0, 6.0];

        let mut model = SimpleLinearRegression::new();
        model.fit(&x, &y).unwrap();

        let preds = model.predict(&x).unwrap();
        assert_abs_diff_eq!(preds, y, epsilon = 1e-6);
    }

    #[test]
    fn test_score_perfect_fit() {
        let x = array![0.0, 1.0, 2.0];
        let y = array![1.0, 3.0, 5.0];

        let mut model = SimpleLinearRegression::new();
        model.fit(&x, &y).unwrap();
        let r2 = model.score(&x, &y).unwrap();
        assert_abs_diff_eq!(r2, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_score_with_noise() {
        let x = array![1.0, 2.0, 3.0, 4.0];
        let y = array![2.0, 4.1, 5.8, 7.9]; // a little noisy

        let mut model = SimpleLinearRegression::new();
        model.fit(&x, &y).unwrap();
        let r2 = model.score(&x, &y).unwrap();
        assert!(r2 > 0.9 && r2 < 1.0);
    }

    #[test]
    fn test_negative_slope() {
        let x = array![1.0, 2.0, 3.0];
        let y = array![5.0, 3.0, 1.0];

        let mut model = SimpleLinearRegression::new();
        model.fit(&x, &y).unwrap();
        assert!(model.slope() < 0.0);
    }

    #[test]
    fn test_predict_output_size() {
        let x = array![1.0, 2.0, 3.0];
        let y = array![2.0, 4.0, 6.0];

        let mut model = SimpleLinearRegression::new();
        model.fit(&x, &y).unwrap();
        let preds = model.predict(&x).unwrap();
        assert_eq!(preds.len(), x.len());
    }

    #[test]
    fn test_predict_before_fit_fails() {
        let model = SimpleLinearRegression::new();
        let x = array![1.0, 2.0];
        assert!(model.predict(&x).is_err());
    }

    #[test]
    fn test_fit_length_mismatch_fails() {
        let x = array![1.0, 2.0];
        let y = array![4.0];
        let mut model = SimpleLinearRegression::new();
        assert!(model.fit(&x, &y).is_err());
    }
}

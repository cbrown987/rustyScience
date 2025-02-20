use num_traits::ToPrimitive;

pub fn l2_loss<L>(prediction: &L, target: &L) -> f64
where
    L: ToPrimitive
{
    let p = prediction.to_f64().unwrap_or(0.0);
    let t = target.to_f64().unwrap_or(0.0);
    (p - t).powi(2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::ToPrimitive;

    #[test]
    fn test_equal_numbers() {
        let prediction = 3.5;
        let target = 3.5;
        let loss = l2_loss(&prediction, &target);
        assert_eq!(loss, 0.0);
    }

    #[test]
    fn test_positive_difference() {
        let prediction = 5.0;
        let target = 2.0;
        let loss = l2_loss(&prediction, &target);
        // (5.0 - 2.0)^2 = 9.0
        assert_eq!(loss, 9.0);
    }

    #[test]
    fn test_negative_numbers() {
        let prediction = -2.0;
        let target = 3.0;
        let loss = l2_loss(&prediction, &target);
        // (-2.0 - 3.0)^2 = 25.0
        assert_eq!(loss, 25.0);
    }

    #[test]
    fn test_integer_values() {
        let prediction = 10;
        let target = 6;
        let loss = l2_loss(&prediction, &target);
        // (10 - 6)^2 = 16.0
        assert_eq!(loss, 16.0);
    }

    // A dummy type that always fails conversion, forcing `to_f64()` to return None.
    #[derive(Debug)]
    struct Dummy;

    impl ToPrimitive for Dummy {
        fn to_i64(&self) -> Option<i64> { None }
        fn to_u64(&self) -> Option<u64> { None }
        fn to_f64(&self) -> Option<f64> { None }
    }

    #[test]
    fn test_dummy_conversion() {
        let prediction = Dummy;
        let target = Dummy;
        let loss = l2_loss(&prediction, &target);
        assert_eq!(loss, 0.0);
    }
}

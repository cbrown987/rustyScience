//! simple linear regression
//!
//! finds best-fit line for y = slope * x + intercept
//!
//! uses basic formula:
//! - slope = cov(x, y) / var(x)
//! - intercept = ȳ - slope * x̄
//!
//! supports predict and r^2 score
//! works with any float type like f32 or f64



use ndarray::{Array1, ArrayBase, Data, Ix1};
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

#[derive(Debug)]
pub struct SimpleLinearRegression<T: Float + FromPrimitive + Debug> {
    intercept: T,
    slope: T,
    fitted: bool,
}

impl<T: Float + FromPrimitive + Debug> SimpleLinearRegression<T> {
    // make new model
    pub fn new() -> Self {
        Self {
            intercept: T::zero(),
            slope: T::zero(),
            fitted: false,
        }
    }

    // train the model
    pub fn fit(&mut self, x: &ArrayBase<impl Data<Elem = T>, Ix1>, y: &ArrayBase<impl Data<Elem = T>, Ix1>) -> Result<(), String> {
        if x.len() != y.len() {
            return Err("lengths don't match".into());
        }

        let xm = x.mean().ok_or("x mean fail")?;
        let ym = y.mean().ok_or("y mean fail")?;

        let mut top = T::zero(); // numerator
        let mut bot = T::zero(); // denominator

        for (xi, yi) in x.iter().zip(y.iter()) {
            top = top + (*xi - xm) * (*yi - ym);
            bot = bot + (*xi - xm).powi(2);
        }

        if bot == T::zero() {
            return Err("div by zero".into());
        }

        self.slope = top / bot;
        self.intercept = ym - self.slope * xm;
        self.fitted = true;
        Ok(())
    }

    // predict y from x
    pub fn predict(&self, x: &ArrayBase<impl Data<Elem = T>, Ix1>) -> Result<Array1<T>, String> {
        if !self.fitted {
            return Err("not fitted".into());
        }
        Ok(x.mapv(|v| self.intercept + self.slope * v))
    }

    // r² score
    pub fn score(&self, x: &ArrayBase<impl Data<Elem = T>, Ix1>, y: &ArrayBase<impl Data<Elem = T>, Ix1>) -> Result<T, String> {
        let preds = self.predict(x)?;
        let ym = y.mean().ok_or("y mean fail")?;

        let mut tot = T::zero(); // total variation
        let mut err = T::zero(); // residual error

        for (yi, ypi) in y.iter().zip(preds.iter()) {
            tot = tot + (*yi - ym).powi(2);
            err = err + (*yi - *ypi).powi(2);
        }

        Ok(T::one() - err / tot)
    }

    // get intercept
    pub fn intercept(&self) -> T {
        self.intercept
    }

    // get slope
    pub fn slope(&self) -> T {
        self.slope
    }
}

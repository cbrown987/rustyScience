use num_traits::ToPrimitive;

pub fn l2_loss<L>(prediction: &L, target: &L) -> f64
where
    L: ToPrimitive
{
    let p = prediction.to_f64().unwrap_or(0.0);
    let t = target.to_f64().unwrap_or(0.0);
    (p - t).powi(2)
}
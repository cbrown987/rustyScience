#[derive(Debug)]
pub enum ModelError {
    EmptyData,
    DimensionMismatch(String),
    InvalidParameter(String),
    PredictionFailure(String),
}

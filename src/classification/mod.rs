pub mod knn;
pub mod tree;
pub mod svc;
pub mod perceptron;

pub use svc::*;

pub use knn::*;
pub use crate::common::knn::{DistanceMetric, WeightType};

pub use tree::*;
pub mod knn;
pub mod tree;
pub mod perceptron;

pub use knn::*;
pub use crate::common::knn::{DistanceMetric, WeightType};

pub use tree::*;
pub use perceptron::*;

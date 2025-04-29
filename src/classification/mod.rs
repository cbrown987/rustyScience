pub mod knn;
pub mod tree;
pub mod svc;
pub mod perceptron;

pub use svc::*;

pub use knn::*;

pub use tree::*;

pub use perceptron::*;

pub use crate::classification::perceptron::MultiClassPerceptron as Perceptron;
//! # rusty_science
//!
//!  An easy to learn and use ML toolkit for rust 
//!
//!
//! ## Installation
//!
//! Add the following to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! crate_name = "0.1.0-alpha"
//! ```
//!
//! Replace `crate_name` with the name of this crate and `version` with the desired version.
//!
//! ## Usage Example
//!
//! ```rust
//! use rusty_science::classification::knn::KNNClassifier;
//! let mut knn = KNNClassifier::new(3);
//! let data = vec![vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 4.0]];
//! let labels = vec![0, 1, 0];
//! knn.fit(data, labels);
//! let prediction = knn.predict(vec![2.5, 3.5]);
//! println!("Predicted label: {}", prediction);
//! ```
//!
//! ## Modules
//!
//! - `classification`: A set of classification models
//! - `clustering`: A set of clustering models
//! - `data`: A set of tools for the manipulation or importing of data
//! - `linear_models`: A set of linear models
//! - `metrics`: Tools to test the output of models
//! - `regression`: A set of regression models
//!
//! ## License
//!
//! Licensed under MIT.
//!
//! ## Contribution
//!
//! Contributions are welcome! Please feel free to submit a pull request or file an issue.
//!
//! ## Acknowledgements
//!
//! If applicable, acknowledge other libraries or individuals that helped in developing this crate.
//! 
//! ## Contributors
//! Cooper Brown, Jack Welsh

// Exposing the linear_models module
pub mod linear_models;

// Exposing the classification module
pub mod classification;

pub mod clustering;

mod common;
pub mod regression;

pub mod metrics;

pub mod data;



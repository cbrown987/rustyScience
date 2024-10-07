// src/lib.rs

// Exposing the linear_models module
pub mod linear_models;

// Exposing the classification module
pub mod classification;

// You can also add utility functions here

pub mod clustering;

// crate specific util functions
pub(crate) mod utils;



#[cfg(test)]
mod tests {
    #[test]
    fn lib_test() {
        unimplemented!()
    }
}

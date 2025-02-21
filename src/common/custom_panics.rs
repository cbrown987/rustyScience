#[macro_export]
macro_rules! panic_untrained {
    ($cond:expr, $model:expr) => {
        if $cond {
            panic!("ERROR: {} must be trained before use.", $model);
        }
    };
}

#[macro_export]
macro_rules! panic_labels_not_binary {
    ($cond:expr, $model:expr) => {
        if $cond {
            panic!("ERROR: Labels must be binary (0,1) for {}",$model);
        }
    };
}



#[macro_export]
macro_rules! panic_missing_coefficients {
    () => {
        panic!("ERROR: Unexpected issue! Model coefficients are missing. Ensure fit() was called.");
    };
}

#[macro_export]
macro_rules! panic_matrix_inversion {
    ($cond:expr, $matrix_name:expr) => {
        if $cond {
            panic!("ERROR: Matrix inversion failed for {}. Ensure it is non-singular.", $matrix_name);
        }
    };
}

#[macro_export]
macro_rules! panic_dimension_mismatch {
    ($cond:expr, $expected:expr, $found:expr) => {
        if $cond {
            panic!(
                "ERROR: Dimension mismatch! Expected: {}, Found: {}",
                $expected, $found
            );
        }
    };
}

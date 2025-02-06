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


use criterion::{Criterion, black_box};
use rusty_science::data::generate_labeled_data;
use rusty_science::regression::{KNNRegression, TreeRegression};

pub(crate) fn benchmark_regression(c: &mut Criterion) {
    let mut group = c.benchmark_group("regression");
    let (data, labels) = generate_labeled_data::<f64, f64>((25, 25));
    let data_cloned = data.clone();
    let labels_cloned = labels.clone();

    group.bench_function("KNNRegression fit", |b| {
        b.iter(|| {
            let mut knn = KNNRegression::<f64, f64>::new(3);
            knn.fit(black_box(data_cloned.clone()), black_box(labels_cloned.clone()));
        })
    });

    group.bench_function("KNNRegression predict", |b| {
        let mut knn = KNNRegression::<f64, f64>::new(3);
        knn.fit(data_cloned.clone(), labels_cloned.clone());
        b.iter(|| {
            knn.predict(black_box(data_cloned[0].clone()));
        })
    });

    group.bench_function("TreeRegression fit", |b| {
        b.iter(|| {
            let mut tree = TreeRegression::<f64>::new();
            tree.fit(black_box(data_cloned.clone()), black_box(labels_cloned.clone()));
        })
    });

    group.bench_function("TreeRegression predict", |b| {
        let mut tree = TreeRegression::<f64>::new();
        tree.fit(data_cloned.clone(), labels_cloned.clone());
        b.iter(|| {
            tree.predict(black_box(data_cloned[0].clone()));
        })
    });

    group.finish();
}
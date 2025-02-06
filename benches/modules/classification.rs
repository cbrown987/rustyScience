use criterion::{Criterion, black_box};
use rusty_science::classification::{KNNClassifier, TreeClassifier};
use rusty_science::classification::perceptron::{BinaryPerceptron, MultiClassPerceptron};
use rusty_science::data::generate_labeled_data;

pub(crate) fn benchmark_classifiers(c: &mut Criterion) {
    let mut group = c.benchmark_group("classification");
    let (data, labels) = generate_labeled_data::<f64, i64>((25, 25));
    let data_cloned = data.clone();
    let labels_cloned = labels.clone();
    
    let (binary_data, binary_labels) = generate_labeled_data::<f64, i64>((25, 1));
    let binary_data_cloned = binary_data.clone();
    let binary_labels_cloned = binary_labels.clone();

    group.bench_function("KNNClassifier fit", |b| {
        b.iter(|| {
            let mut knn = KNNClassifier::<f64, i64>::new(3);
            knn.fit(black_box(data_cloned.clone()), black_box(labels_cloned.clone()));
        })
    });

    group.bench_function("KNNClassifier predict", |b| {
        let mut knn = KNNClassifier::<f64, i64>::new(3);
        knn.fit(data_cloned.clone(), labels_cloned.clone());
        b.iter(|| {
            knn.predict(black_box(data_cloned[0].clone()));
        })
    });

    group.bench_function("TreeClassifier fit", |b| {
        b.iter(|| {
            let mut tree = TreeClassifier::<f64, i64>::new();
            tree.fit(black_box(data_cloned.clone()), black_box(labels_cloned.clone()));
        })
    });

    group.bench_function("TreeClassifier predict", |b| {
        let mut tree = TreeClassifier::<f64, i64>::new();
        tree.fit(data_cloned.clone(), labels_cloned.clone());
        b.iter(|| {
            tree.predict(black_box(data_cloned[0].clone()));
        })
    });

    group.bench_function("MultiClassPerceptron fit", |b| {
        b.iter(|| {
            let mut perceptron = MultiClassPerceptron::<f64, i64>::new();
            perceptron.fit(data_cloned.clone(), labels_cloned.clone());        })
    });

    group.bench_function("MultiClassPerceptron predict", |b| {
        let mut perceptron = MultiClassPerceptron::<f64, i64>::new();
        perceptron.fit(data_cloned.clone(), labels_cloned.clone());
        b.iter(|| {
            perceptron.predict(black_box(data_cloned[0].clone()));
        })
    });

    group.bench_function("BinaryPerceptron fit", |b| {
        b.iter(|| {
            let mut perceptron = BinaryPerceptron::<f64, i64>::new();
            perceptron.fit(binary_data_cloned.clone(), binary_labels_cloned.clone());       
        })
    });

    group.bench_function("BinaryPerceptron predict", |b| {
        let mut perceptron = BinaryPerceptron::<f64, i64>::new();
        perceptron.fit(binary_data_cloned.clone(), binary_labels_cloned.clone());
        b.iter(|| {
            perceptron.predict(black_box(binary_data_cloned[0].clone()));
        })
    });
}
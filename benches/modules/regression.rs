use criterion::{black_box, BenchmarkId, Criterion, Throughput};
use rusty_science::data::generate_labeled_data;
use rusty_science::regression::{KNNRegression, TreeRegression, WeightType, DistanceMetric};
use std::time::Duration;

pub(crate) fn benchmark_regression(c: &mut Criterion) {
    let mut group = c.benchmark_group("regression");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));
    
    // Test with varying dataset sizes to understand scaling behavior
    let sizes = [(10, 5), (50, 10), (100, 20), (250, 50)];
    
    for size in sizes.iter() {
        let size_id = format!("{}x{}", size.0, size.1);
        group.throughput(Throughput::Elements((size.0 * size.1) as u64));
        
        // Generate data once per size
        let (data, labels) = generate_labeled_data::<f64, f64>(*size);
        
        // KNN benchmarks with different configurations
        for &k in &[1, 3, 7, 15] {
            // Benchmark KNN fitting
            group.bench_with_input(
                BenchmarkId::new("KNN_fit", format!("k={},size={}", k, size_id)), 
                &(data.clone(), labels.clone()),
                |b, (data, labels)| {
                    b.iter(|| {
                        let mut knn = KNNRegression::<f64, f64>::new(k);
                        knn.fit(black_box(data.clone()), black_box(labels.clone()))
                    })
                }
            );
            
            // Create model once for prediction benchmarks
            let mut knn = KNNRegression::<f64, f64>::new(k);
            knn.fit(data.clone(), labels.clone()).unwrap();
            
            // Benchmark single prediction
            group.bench_with_input(
                BenchmarkId::new("KNN_predict_single", format!("k={},size={}", k, size_id)),
                &data[0],
                |b, sample| {
                    b.iter(|| knn.predict(black_box(sample.clone())))
                }
            );
            
            // Benchmark batch prediction (predict all samples)
            group.bench_with_input(
                BenchmarkId::new("KNN_predict_batch", format!("k={},size={}", k, size_id)),
                &data,
                |b, test_data| {
                    b.iter(|| {
                        for sample in test_data.iter() {
                            black_box(knn.predict(sample.clone()));
                        }
                    })
                }
            );
        }
        
        // Different weight types for KNN
        for weight_type in &[WeightType::Uniform, WeightType::Distance] {
            let mut knn = KNNRegression::<f64, f64>::new(5);
            knn.set_weight_type(weight_type.clone());
            knn.fit(data.clone(), labels.clone()).unwrap();
            
            group.bench_with_input(
                BenchmarkId::new("KNN_weight_type", format!("{:?}_size={}", weight_type, size_id)),
                &data,
                |b, test_data| {
                    b.iter(|| {
                        for sample in test_data.iter() {
                            black_box(knn.predict(sample.clone()));
                        }
                    })
                }
            );
        }
        
        // Different distance metrics for KNN
        for metric in &[DistanceMetric::Euclidean, DistanceMetric::Manhattan] {
            let mut knn = KNNRegression::<f64, f64>::new(5);
            knn.set_distance_metric(metric.clone());
            knn.fit(data.clone(), labels.clone()).unwrap();
            
            group.bench_with_input(
                BenchmarkId::new("KNN_distance_metric", format!("{:?}_size={}", metric, size_id)),
                &data,
                |b, test_data| {
                    b.iter(|| {
                        for sample in test_data.iter() {
                            black_box(knn.predict(sample.clone()));
                        }
                    })
                }
            );
        }
        
        // Tree regression benchmarks
        group.bench_with_input(
            BenchmarkId::new("Tree_fit", format!("size={}", size_id)),
            &(data.clone(), labels.clone()),
            |b, (data, labels)| {
                b.iter(|| {
                    let mut tree = TreeRegression::<f64>::new();
                    tree.fit(black_box(data.clone()), black_box(labels.clone()))
                })
            }
        );
        
        // Create model once for prediction benchmarks
        let mut tree = TreeRegression::<f64>::new();
        tree.fit(data.clone(), labels.clone());
        
        // Benchmark single prediction
        group.bench_with_input(
            BenchmarkId::new("Tree_predict_single", format!("size={}", size_id)),
            &data[0],
            |b, sample| {
                b.iter(|| tree.predict(black_box(sample.clone())))
            }
        );
        
        // Benchmark batch prediction
        group.bench_with_input(
            BenchmarkId::new("Tree_predict_batch", format!("size={}", size_id)),
            &data,
            |b, test_data| {
                b.iter(|| {
                    for sample in test_data.iter() {
                        black_box(tree.predict(sample.clone()));
                    }
                })
            }
        );
        
        // Tree vs KNN comparison for same size
        // First create optimized models
        let mut optimized_knn = KNNRegression::<f64, f64>::new(5);
        optimized_knn.fit(data.clone(), labels.clone()).unwrap();
        
        let mut optimized_tree = TreeRegression::<f64>::new();
        optimized_tree.fit(data.clone(), labels.clone());
        
        // Test samples from outside training set for better real-world comparison
        let (test_data, _) = generate_labeled_data::<f64, f64>((10, size.1)); 
        
        // Benchmark comparison
        group.bench_with_input(
            BenchmarkId::new("KNN_vs_Tree", format!("size={}", size_id)),
            &test_data,
            |b, test_samples| {
                b.iter_with_large_drop(|| {
                    // KNN predictions
                    let knn_predictions: Vec<f64> = test_samples.iter()
                        .map(|sample| optimized_knn.predict(sample.clone()))
                        .collect();
                    
                    // Tree predictions
                    let tree_predictions: Vec<f64> = test_samples.iter()
                        .map(|sample| optimized_tree.predict(sample.clone()))
                        .collect();
                    
                    (knn_predictions, tree_predictions)
                })
            }
        );
    }
    
    group.finish();
}

// Additional benchmark function for stress testing with larger datasets
pub(crate) fn benchmark_regression_large_scale(c: &mut Criterion) {
    let mut group = c.benchmark_group("regression_large_scale");
    group.sample_size(10); // Fewer samples for large scale tests
    group.warm_up_time(Duration::from_millis(100));
    group.measurement_time(Duration::from_secs(5));
    
    // Large datasets
    let size = (500, 100);
    let (data, labels) = generate_labeled_data::<f64, f64>(size);
    
    // KNN with smaller k values (more efficient)
    for &k in &[1, 5] {
        let title = format!("KNN_k={}_large", k);
        group.bench_function(&title, |b| {
            b.iter_with_large_drop(|| {
                let mut knn = KNNRegression::<f64, f64>::new(k);
                knn.fit(data.clone(), labels.clone()).unwrap();
                knn
            })
        });
    }
    
    // Tree regression
    group.bench_function("Tree_large", |b| {
        b.iter_with_large_drop(|| {
            let mut tree = TreeRegression::<f64>::new();
            tree.fit(data.clone(), labels.clone());
            tree
        })
    });
    
    group.finish();
}

// Register all benchmark functions
pub fn register_regression_benchmarks(c: &mut Criterion) {
    benchmark_regression(c);
    benchmark_regression_large_scale(c);
}
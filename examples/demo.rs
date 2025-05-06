fn main() {
    println!("RUSTY SCIENCE FULL DEMO START\n");

    println!("⚡ Running Linear Models Demo...\n");
    demo_linear_models::main();

    println!("⚡ Running Regression Demo...\n");
    demo_regression::main();

    println!("⚡ Running Classification Demo...\n");
    demo_classification::main();

    println!("⚡ Running Clustering Demo...\n");
    demo_clustering::main();

    println!("⚡ Running Benchmarks...\n");
    demo_benchmarks::main();

    println!("\nRUSTY SCIENCE FULL DEMO END");
}

mod demo_linear_models;
mod demo_regression;
mod demo_classification;
mod demo_clustering;
mod demo_benchmarks;
// cargo run --example demo_linear_models
// cargo run --example demo_regression
// cargo run --example demo_classification
// cargo run --example demo_clustering
// cargo run --example demo_benchmarks


//TODO
// - Benchmarks
// - Fix Regression imports
// - DBSCAN
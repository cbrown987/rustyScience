use rusty_science::data::datasets::load_iris;
use rusty_science::neural_network::model::MultiLayerPerceptron;
use rusty_science::neural_network::normalize::normalize;

fn main() {
    println!("Training MLP on Iris dataset...");
    

    // load Iris data
    let iris = load_iris();
    let (data, raw_labels) = iris.to_numerical_labels();
    let data = normalize(&data);
    // convert labels from <i64> to <usize>
    let labels: Vec<usize> = raw_labels.iter().map(|x| *x as usize).collect();

    // Create a 3-layer MLP: 4 inputs -> hidden  -> 3 outputs
    let mut mlp = MultiLayerPerceptron::<f64>::new(&[4, 5, 3], true); // true = use sigmoid. false = relu

    // Train the model
    mlp.fit(&data, &labels, 100, 0.1);

    println!("\nEvaluating predictions:"); //data wrapping for ease of read
    for (x, true_label) in data.iter().zip(labels.iter()).take(5) {
        let output = mlp.forward(x);
        let predicted_label = output
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(idx, _)| idx)
            .unwrap();

        println!(
            "Input: {:?} -> Predicted: {} | Output: {:?} | True: {}",
            x, predicted_label, output, true_label
        );
    }
}

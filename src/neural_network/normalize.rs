/// Normalize a 2D dataset (Vec<Vec<f64>>)
/// - Each feature column is zero-centered and scaled to unit variance
/// - Returns a new normalized dataset


pub fn normalize(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let num_features = data[0].len();
    let num_samples = data.len();

    let mut means = vec![0.0; num_features];
    let mut stds = vec![0.0; num_features];

    //compute mean for each feature
    for i in 0..num_features {
        for row in data {
            means[i] += row[i];
        }
        means[i] /= num_samples as f64;
    }

    //compute std dev for each feature
    for i in 0..num_features {
        for row in data {
            let diff = row[i] - means[i];
            stds[i] += diff * diff;
        }
        stds[i] = (stds[i] / num_samples as f64).sqrt();
        if stds[i] == 0.0 {
            stds[i] = 1.0; //prevent /0
        }
    }

    //normalize dataset
    data.iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(i, val)| (val - means[i]) / stds[i])
                .collect()
        })
        .collect()
}
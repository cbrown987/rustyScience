use std::collections::BinaryHeap;
use crate::utils::euclidean_distance;
pub struct KMeans {
    k: usize,
    features: Vec<Vec<f64>>,
    labels: Vec<f64>,
}

impl KMeans {
    pub fn new(k: usize) -> KMeans {
        KMeans{
            k,
            features: vec![],
            labels: vec![],
        }
    }
    pub fn fit(&mut self, features: Vec<Vec<f64>>, labels: Vec<f64>) {
        self.features = features;
        self.labels = labels;
    }
    pub fn predict(&self, input: &[f64]) -> f64 {
        let mut distances = BinaryHeap::new();

        // Calculate distances to each training point
        for (i, feature) in self.features.iter().enumerate() {
            let distance = euclidean_distance(input, feature);
            distances.push((distance, self.labels[i]));
        }

        let mut neighbors = distances
            .into_sorted_vec()
            .into_iter()
            .take(self.k)
            .map(|(_, label)| label)
            .collect::<Vec<f64>>();

        neighbors.sort_unstable();
        let most_common:(f64, usize) = neighbors.iter().fold((0f64, 0), |(most_common, count), &label| {
            let label_count = neighbors.iter().filter(|&&x| x == label).count();
            if label_count > count {
                (label, label_count)
            } else {
                (most_common, count)
            }
        });

        most_common.0
    }

}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        unimplemented!()
    }
}
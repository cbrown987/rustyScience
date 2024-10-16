use std::ops::{Add, Sub, Div};
use std::convert::From;
use num::complex::ComplexFloat;
use rand::prelude::*;
use crate::common::utils::euclidean_distance;

struct KMeansClassifier{
    k: usize,
    data: Vec<Vec<f64>>,
    labels: Vec<f64>,
}
impl KMeansClassifier{
    fn new(k: usize) -> Self{
        Self{
            k,
            data: vec![vec![]],
            labels: vec![],
        }
    }

    fn fit(&mut self, data: Vec<Vec<f64>>, labels: Vec<f64>){
        self.data = data;
        self.labels = labels;
    }

    fn distance(&self, x: &[f64], y: &[f64]) -> f64{
        euclidean_distance(x, y)
    }
    fn predict(&self, target: Vec<f64>) -> f64{
        let target_slice = target.as_slice();
        let mut distances: Vec<(f64, f64)> = self.data
            .iter()
            .zip(self.labels.iter())
            .map(|(point, &label)| (self.distance(point.as_slice(), target_slice), label))
            .collect();

        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        distances[0].1
    }
}

#[cfg(test)]
mod tests {
    use crate::classification::k_means::KMeansClassifier;
    #[test]
    fn it_works() {
        let data = vec![
            vec![1.0, 2.0],
            vec![1.5, 1.8],
            vec![5.0, 8.0],
            vec![8.0, 8.0],
        ];
        let labels = vec![
            1.0,
            1.0,
            1.0,
            2.0
        ];
        let target = vec![7.0,8.0];
        let mut kmeans = KMeansClassifier::new( 0);
        kmeans.fit(data, labels);
        let prediction = kmeans.predict(target);

        assert_eq!(prediction, 2.0);
    }
}
use num_traits::{Num, ToPrimitive};
use rand::distributions::{Distribution, Uniform};
use rand::distributions::uniform::SampleUniform;

pub fn generate_labeled_data<D, L>(shape: (usize, usize)) -> (Vec<Vec<D>>, Vec<L>)
where
    D: Num + Copy + Clone + PartialOrd + ToPrimitive + SampleUniform,
    L: Num + Copy + Clone + PartialOrd + ToPrimitive + SampleUniform,
{
    let (rows, _) = shape;
    let mut rng = rand::thread_rng();

    let data = generate_unlabeled_data::<D>(shape);
    
    let label_dist = Uniform::new(L::zero(), L::one());
    let labels: Vec<L> = (0..rows).map(|_| label_dist.sample(&mut rng)).collect();

    (data, labels)
}

pub fn generate_unlabeled_data<D>(shape: (usize, usize)) -> Vec<Vec<D>>
where
    D: Num + Copy + Clone + PartialOrd + ToPrimitive + SampleUniform,
{
    let (rows, cols) = shape;
    let mut rng = rand::thread_rng();

    let data_dist = Uniform::new(D::zero(), D::one());

    let data: Vec<Vec<D>> = (0..rows)
        .map(|_| (0..cols).map(|_| data_dist.sample(&mut rng)).collect())
        .collect();
    
    data
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_generate_labeled_data() {
        let (data, labels) = generate_labeled_data::<f64, f64>((100,100));
        assert_eq!(data.len(), 100);
        assert_eq!(labels.len(), 100);
    }
    
    #[test]
    fn test_generate_unlabeled_data() {
        let data = generate_unlabeled_data::<f64>((100,100));
        assert_eq!(data.len(), 100);
    }
}
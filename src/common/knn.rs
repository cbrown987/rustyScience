use std::cmp::Ordering;
use num_traits::{Num, ToPrimitive};
use crate::common::utils::{euclidean_distance, manhattan_distance};

#[allow(dead_code)]
pub(crate) struct Neighbor<D, L> {
    pub(crate) point: Vec<D>,
    pub(crate) label: Option<L>,
    pub (crate) distance_to_target: f64,
}

fn _handle_labels<L>(i: usize, data_labels: &Option<Vec<L>>) -> Option<L>
where
    L: Copy + Clone,
{
    data_labels.as_ref().map(|labels| labels[i])
}

fn _process_neighbors<D, L>(
    neighbor_indices: &[usize],
    data: &[Vec<D>],
    distances: &[(usize, f64)],
    calculate_distance: bool,
    data_labels: &Option<Vec<L>>,
) -> Vec<Neighbor<D, L>>
where
    D: Num + Copy + Clone,
    L: Copy + Clone,
{
    neighbor_indices
        .iter()
        .enumerate()
        .map(|(_index, &i)| {
            let point = data[i].clone();
            let label = _handle_labels(i, data_labels);
            let distance_to_target = if calculate_distance {
                distances
                    .iter()
                    .find(|&&(idx, _)| idx == i)
                    .map(|&(_, dist)| dist)
                    .unwrap_or(0.0)
            } else {
                0.0
            };
            Neighbor {
                point,
                label,
                distance_to_target,
            }
        })
        .collect()
}

/// Main neighbors function
pub(crate) fn neighbors<D, L>(
    data: Vec<Vec<D>>,
    data_labels: Option<Vec<L>>,
    target_point: Option<Vec<D>>,
    n_neighbors: usize,
    distance_metric: String,
    calculate_distance: bool,
) -> Vec<Neighbor<D, L>>
where
    D: Num + ToPrimitive + Copy + PartialOrd,
    L: Copy + Clone,
{
    let distance_fn: Box<dyn Fn(&[D], &[D]) -> f64> = match distance_metric.to_lowercase().as_str() {
        "euclidean" => Box::new(euclidean_distance),
        "manhattan" => Box::new(manhattan_distance),
        _ => panic!("Unknown distance metric"),
    };

    if let Some(target_point) = target_point {
        let mut distances: Vec<(usize, f64)> = data
            .iter()
            .enumerate()
            .map(|(i, point)| (i, distance_fn(&target_point, point)))
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        let neighbor_indices: Vec<usize> = distances
            .iter()
            .take(n_neighbors)
            .map(|(i, _)| *i)
            .collect();

        _process_neighbors(
            &neighbor_indices,
            &data,
            &distances,
            calculate_distance,
            &data_labels,
        )
    } else {
        let mut neighbors_for_all_points = Vec::new();
        for (idx, point) in data.iter().enumerate() {
            let mut distances: Vec<(usize, f64)> = data
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != idx)
                .map(|(i, other_point)| (i, distance_fn(point, other_point)))
                .collect();

            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

            let neighbor_indices: Vec<usize> = distances
                .iter()
                .take(n_neighbors)
                .map(|(i, _)| *i)
                .collect();

            let neighbors = _process_neighbors(
                &neighbor_indices,
                &data,
                &distances,
                calculate_distance,
                &data_labels,
            );

            neighbors_for_all_points.extend(neighbors);
        }
        neighbors_for_all_points
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::test_utils::{create_data_labeled, create_data_unlabeled};

    #[test]
    fn test_neighbors_euclidean_with_distance() {
        let data = create_data_unlabeled().get("small_data").unwrap().clone();

        let target_point = Some(vec![1.0, 1.0]);
        let n_neighbors = 2;
        let distance_metric = String::from("euclidean");
        let calculate_distance = true;

        // Adjusted the function call to match the new signature
        let result = neighbors::<f64, f64>(
            data,
            None, // No labels provided
            target_point,
            n_neighbors,
            distance_metric,
            calculate_distance,
        );

        // Check that the correct neighbors are returned with distances
        assert_eq!(result.len(), 2);

        // The first neighbor should be the point itself with distance 0.0
        assert_eq!(result[0].point, vec![1.0, 1.0]);
        assert_eq!(result[0].distance_to_target, 0.0);

        // The second neighbor should be (0.0, 0.0)
        assert_eq!(result[1].point, vec![0.0, 0.0]);
        // Calculate the expected distance
        let expected_distance = ((1.0f64 - 0.0).powi(2) + (1.0f64 - 0.0).powi(2)).sqrt();
        assert!((result[1].distance_to_target - expected_distance).abs() < 1e-5);
    }

    #[test]
    fn test_neighbors_euclidean_without_distance() {
        let data = create_data_unlabeled().get("small_data").unwrap().clone();

        let target_point = Some(vec![1.0, 1.0]);
        let n_neighbors = 2;
        let distance_metric = String::from("euclidean");
        let calculate_distance = false;

        let result = neighbors::<f64, f64>(
            data,
            None,
            target_point,
            n_neighbors,
            distance_metric,
            calculate_distance,
        );

        // Check that the correct neighbors are returned without distances
        assert_eq!(result.len(), 2);

        // The first neighbor should be the point itself with distance 0.0
        assert_eq!(result[0].point, vec![1.0, 1.0]);
        // Since calculate_distance is false, distance_to_target should be 0.0 by default
        assert_eq!(result[0].distance_to_target, 0.0);

        // The second neighbor should be (0.0, 0.0)
        assert_eq!(result[1].point, vec![0.0, 0.0]);
        // Distance is not calculated, so distance_to_target should be 0.0
        assert_eq!(result[1].distance_to_target, 0.0);
    }

    #[test]
    fn test_neighbors_manhattan_with_distance() {
        let data = create_data_unlabeled().get("small_data").unwrap().clone();

        let target_point = Some(vec![1.0, 1.0]);
        let n_neighbors = 2;
        let distance_metric = String::from("manhattan");
        let calculate_distance = true;

        let result = neighbors::<f64, f64>(
            data,
            None,
            target_point,
            n_neighbors,
            distance_metric,
            calculate_distance,
        );

        // Check that the correct neighbors are returned with Manhattan distances
        assert_eq!(result.len(), 2);

        // The first neighbor should be the point itself with distance 0.0
        assert_eq!(result[0].point, vec![1.0, 1.0]);
        assert_eq!(result[0].distance_to_target, 0.0);

        // The second neighbor should be (0.0, 0.0)
        assert_eq!(result[1].point, vec![0.0, 0.0]);
        // Calculate the expected Manhattan distance
        let expected_distance = (1.0f64 - 0.0).abs() + (1.0f64 - 0.0).abs(); // Should be 2.0
        assert!((result[1].distance_to_target - expected_distance).abs() < 1e-5);
    }

    #[test]
    fn test_neighbors_manhattan_without_distance() {
        let data = create_data_unlabeled().get("small_data").unwrap().clone();

        let target_point = Some(vec![1.0, 1.0]);
        let n_neighbors = 2;
        let distance_metric = String::from("manhattan");
        let calculate_distance = false;

        let result = neighbors::<f64, f64>(
            data,
            None,
            target_point,
            n_neighbors,
            distance_metric,
            calculate_distance,
        );

        // Check that the correct neighbors are returned without distances
        assert_eq!(result.len(), 2);

        // The first neighbor should be the point itself
        assert_eq!(result[0].point, vec![1.0, 1.0]);
        // Distance is not calculated, so distance_to_target should be 0.0
        assert_eq!(result[0].distance_to_target, 0.0);

        // The second neighbor should be (0.0, 0.0)
        assert_eq!(result[1].point, vec![0.0, 0.0]);
        assert_eq!(result[1].distance_to_target, 0.0);
    }

    #[test]
    #[should_panic(expected = "Unknown distance metric")]
    fn test_invalid_distance_metric() {
        let data = create_data_unlabeled().get("small_data").unwrap().clone();

        let target_point = Some(vec![1.0, 1.0]);
        let n_neighbors = 2;
        let distance_metric = String::from("invalid_metric");
        let calculate_distance = false;

        // This should panic because the distance metric is invalid
        neighbors::<f64, f64>(
            data,
            None,
            target_point,
            n_neighbors,
            distance_metric,
            calculate_distance,
        );
    }

    #[test]
    fn test_neighbors_with_labels_with_distance() {
        let target_point = Some(vec![1.0, 1.0]);
        let n_neighbors = 3;
        let distance_metric = String::from("euclidean");
        let calculate_distance = true;

        if let Some(dataset) = create_data_labeled().get("small_data") {
            let result = neighbors(
                dataset.data.clone(),
                Some(dataset.labels.clone()),
                target_point,
                n_neighbors,
                distance_metric,
                calculate_distance,
            );

            // Check that the correct neighbors are returned with distances
            assert_eq!(result.len(), 3);

            // The first neighbor should be the point itself with label and distance 0.0
            assert_eq!(result[0].point, vec![1.0, 1.0]);
            assert_eq!(result[0].label, Some(1i64)); // Assuming labels are of type f64
            assert_eq!(result[0].distance_to_target, 0.0);

            // The second neighbor
            assert_eq!(result[1].label, Some(1)); // Check the label
            // Compute expected distance
            let expected_distance = ((1.0f64 - 0.0).powi(2) + (1.0f64 - 0.0).powi(2)).sqrt();
            assert!((result[1].distance_to_target - expected_distance).abs() < 1e-5);

            // The third neighbor
            assert_eq!(result[2].label, Some(10));
        } else {
            panic!("Failed to create dataset");
        }
    }
}

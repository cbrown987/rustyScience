use crate::common::utils::{euclidean_distance, manhattan_distance};

pub(crate) fn neighbors(data: Vec<Vec<f64>>, data_labels_categorical: Option<Vec<i64>>, 
                        data_labels_regression: Option<Vec<f64>>, target_point: Option<Vec<f64>>,
                        n_neighbors: usize, distance_metric:String, calculate_distance: bool) -> Vec<Vec<f64>> {

    fn _handle_labels(mut neighbor: Vec<f64>, i: usize, data_labels_regression: &Option<Vec<f64>>, 
                      data_labels_categorical: &Option<Vec<i64>>, ) -> Vec<f64> {
        if let Some(labels) = data_labels_regression {
            let data_label = labels[i];
            neighbor.push(data_label);
        } else if let Some(labels) = data_labels_categorical {
            let data_label = labels[i];
            neighbor.push(data_label as f64);
        }
        neighbor
    }

    fn _process_neighbors(neighbor_indices: &[usize], data: &[Vec<f64>], distances: &[(usize, f64)],
                          calculate_distance: bool, data_labels_regression: &Option<Vec<f64>>, 
                          data_labels_categorical: &Option<Vec<i64>>) -> Vec<Vec<f64>> {
        neighbor_indices
            .iter()
            .enumerate()
            .map(|(index, &i)| {
                let mut neighbor = data[i].clone();
                if calculate_distance {
                    neighbor.push(distances[index].1);
                }
                _handle_labels(neighbor, i, data_labels_regression, data_labels_categorical)
            })
            .collect()
    }
    
    let distance_fn: Box<dyn Fn(&[f64], &[f64]) -> f64> = match distance_metric.as_str() {
        "euclidean" => Box::new(euclidean_distance),
        "manhattan" => Box::new(manhattan_distance),
        _ => panic!("Unknown distance metric"),
    };

    // If no target_point is provided, find neighbors for all points in the dataset
    if target_point.is_none() {
        // TODO: Implement
        unimplemented!("Handling neighbors for each point not yet implemented.");
    }
    
    if !data_labels_categorical.is_none() & !data_labels_regression.is_none() {
        panic!("Only one type of data labels is allowed")
    }

    let target_point = target_point.unwrap();

    let mut distances: Vec<(usize, f64)> = data
        .iter()
        .enumerate()
        .map(|(i, point)| (i, distance_fn(&target_point, point)))
        .collect();

    // Sort by distance and select the closest n_neighbors
    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    // Extract the indices of the n_neighbors closest points
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
        &data_labels_regression,
        &data_labels_categorical,
    )
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

        let result = neighbors(data, None, None, target_point, n_neighbors, distance_metric, calculate_distance);

        // Check that the correct neighbors are returned with distances
        // The neighbors should be (1.0, 1.0) and (0.0, 0.0) with distances
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], vec![1.0, 1.0, 0.0]); // The first point is itself, distance 0
        assert_eq!(result[1][0..2], [0.0, 0.0]);    // The second point is (0.0, 0.0)
    }

    #[test]
    fn test_neighbors_euclidean_without_distance() {
        let data = create_data_unlabeled().get("small_data").unwrap().clone();

        let target_point = Some(vec![1.0, 1.0]);
        let n_neighbors = 2;
        let distance_metric = String::from("euclidean");
        let calculate_distance = false;

        let result = neighbors(data, None, None, target_point,n_neighbors, distance_metric, calculate_distance);

        // Check that the correct neighbors are returned without distances
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], vec![1.0, 1.0]);      // The first point is itself
        assert_eq!(result[1], vec![0.0, 0.0]);      // The second point is (0.0, 0.0)
    }

    #[test]
    fn test_neighbors_manhattan_with_distance() {
        let data = create_data_unlabeled().get("small_data").unwrap().clone();

        let target_point = Some(vec![1.0, 1.0]);
        let n_neighbors = 2;
        let distance_metric = String::from("manhattan");
        let calculate_distance = true;

        let result = neighbors(data, None, None, target_point, n_neighbors, distance_metric, calculate_distance);

        // Check that the correct neighbors are returned with Manhattan distances
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], vec![1.0, 1.0, 0.0]); // The first point is itself, distance 0
        assert_eq!(result[1][0..2], [0.0, 0.0]);    // The second point is (0.0, 0.0)
        assert!(result[1][2] - 2.0 < 1e-5);         // The Manhattan distance should be 2.0
    }

    #[test]
    fn test_neighbors_manhattan_without_distance() {
        let data = create_data_unlabeled().get("small_data").unwrap().clone();

        let target_point = Some(vec![1.0, 1.0]);
        let n_neighbors = 2;
        let distance_metric = String::from("manhattan");
        let calculate_distance = false;

        let result = neighbors(data, None, None, target_point, n_neighbors, distance_metric, calculate_distance);

        // Check that the correct neighbors are returned without distances
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], vec![1.0, 1.0]);      // The first point is itself
        assert_eq!(result[1], vec![0.0, 0.0]);      // The second point is (0.0, 0.0)
    }

    #[test]
    #[should_panic]
    fn test_invalid_distance_metric() {
        let data = create_data_unlabeled().get("small_data").unwrap().clone();

        let target_point = Some(vec![1.0, 1.0]);
        let n_neighbors = 2;
        let distance_metric = String::from("invalid_metric");
        let calculate_distance = false;

        // This should panic because the distance metric is invalid
        neighbors(data, None, None, target_point, n_neighbors, distance_metric, calculate_distance);
    }
    
    #[test]
    fn test_neighbors_with_labels_with_distance(){
        
        let target_point = Some(vec![1.0, 1.0]);
        let n_neighbors = 3;
        let distance_metric = String::from("euclidean");
        let calculate_distance = true;
        
        if let Some(dataset) = create_data_labeled().get("small_data") {
            let result = neighbors(dataset.data.clone(), Option::from(dataset.labels.clone()), None, target_point, n_neighbors, distance_metric, calculate_distance);
            // Check that the correct neighbors are returned with distances
            // The neighbors should be (1.0, 1.0) and (0.0, 0.0) with distances
            assert_eq!(result.len(), 3);
            assert_eq!(result[0], vec![1.0, 1.0, 0.0, 1.0]); // The first point is itself, label
            assert_eq!(result[1][3], 1.0);    // Test the label of second point is correct
            assert_eq!(result[2][3], 10.0);
            
        } else { assert!(false, "Failed to create dataset"); }

       
    }
}

use num_traits::{Num, ToPrimitive};

pub(crate) struct Node<D, L> {
    pub(crate) is_leaf: bool,
    pub(crate) prediction: Option<L>,
    pub(crate) feature_index: Option<usize>,
    pub(crate) threshold: Option<D>,
    pub(crate) left: Option<Box<Node<D, L>>>,
    pub(crate) right: Option<Box<Node<D, L>>>,
}

#[derive(Debug, Clone)]
pub(crate) struct Instance<D, L> {
    pub(crate) data: Vec<D>,
    pub(crate) label: L,
}

pub(crate) fn gini_impurity<D, L>(instances: &[Instance<D, L>]) -> f64
where
    L: PartialEq + Clone,
{
    let mut label_counts: Vec<(L, usize)> = Vec::new();

    for instance in instances {
        let mut found = false;
        for (label, count) in &mut label_counts {
            if *label == instance.label {
                *count += 1;
                found = true;
                break;
            }
        }
        if !found {
            label_counts.push((instance.label.clone(), 1));
        }
    }

    let total_instances = instances.len() as f64;

    if total_instances == 0.0 {
        return 0.0; // Handle the case with no instances
    }

    // Calculate the Gini impurity
    let impurity = label_counts
        .iter()
        .map(|(_, count)| {
            let probability = *count as f64 / total_instances;
            probability * probability
        })
        .sum::<f64>();

    1.0 - impurity
}


pub(crate) fn find_best_split<D, L>(
    instances: &[Instance<D, L>]) -> Option<(usize, Option<D>, Vec<Instance<D, L>>, Vec<Instance<D, L>>)> 
where
    D: Num + Copy + Clone + PartialOrd + ToPrimitive,
    L: Num + Copy + Clone + PartialOrd + ToPrimitive,
{
    if instances.is_empty() {
        return None;
    }

    let num_features = instances[0].data.len();
    let mut best_feature = 0;
    let mut best_threshold: Option<D> = None;
    let mut best_impurity = f64::INFINITY;
    let mut best_left = Vec::new();
    let mut best_right = Vec::new();

    for feature_index in 0..num_features {
        // Collect and sort unique thresholds
        let mut thresholds: Vec<D> = instances
            .iter()
            .map(|inst| inst.data[feature_index])
            .collect();
        thresholds.sort_by(|a, b| a.partial_cmp(b).unwrap());
        thresholds.dedup();

        for &threshold in &thresholds {
            let (left, right): (Vec<Instance<D, L>>, Vec<Instance<D, L>>) = instances
                .iter()
                .cloned()
                .partition(|inst| inst.data[feature_index] <= threshold);

            if left.is_empty() || right.is_empty() {
                continue;
            }

            let impurity_left = gini_impurity(&left);
            let impurity_right = gini_impurity(&right);

            let total_len = instances.len() as f64;
            let impurity = (left.len() as f64 * impurity_left
                + right.len() as f64 * impurity_right)
                / total_len;

            if impurity < best_impurity {
                best_impurity = impurity;
                best_feature = feature_index;
                best_threshold = Some(threshold);
                best_left = left;
                best_right = right;
            }
        }
    }

    if best_impurity == f64::INFINITY {
        None
    } else {
        Some((best_feature, best_threshold, best_left, best_right))
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gini_impurity_empty() {
        let instances: Vec<Instance<f64, usize>> = vec![];
        let impurity = gini_impurity(&instances);
        assert_eq!(impurity, 0.0);
    }

    #[test]
    fn test_gini_impurity_same_label() {
        let instances = vec![
            Instance {
                data: vec![1.0],
                label: 0,
            },
            Instance {
                data: vec![2.0],
                label: 0,
            },
            Instance {
                data: vec![3.0],
                label: 0,
            },
        ];
        let impurity = gini_impurity(&instances);
        assert_eq!(impurity, 0.0);
    }

    #[test]
    fn test_gini_impurity_two_labels_equal() {
        let instances = vec![
            Instance {
                data: vec![1.0],
                label: 0,
            },
            Instance {
                data: vec![2.0],
                label: 1,
            },
            Instance {
                data: vec![3.0],
                label: 0,
            },
            Instance {
                data: vec![4.0],
                label: 1,
            },
        ];
        let impurity = gini_impurity(&instances);
        assert_eq!(impurity, 0.5);
    }

    #[test]
    fn test_gini_impurity_two_labels_unequal() {
        let instances = vec![
            Instance {
                data: vec![1.0],
                label: 0,
            },
            Instance {
                data: vec![2.0],
                label: 0,
            },
            Instance {
                data: vec![3.0],
                label: 0,
            },
            Instance {
                data: vec![4.0],
                label: 1,
            },
        ];
        let impurity = gini_impurity(&instances);
        assert!((impurity - 0.375).abs() < 1e-6);
    }

    #[test]
    fn test_find_best_split_empty() {
        let instances: Vec<Instance<f64, usize>> = vec![];
        let result = find_best_split(&instances);
        assert!(result.is_none());
    }

    #[test]
    fn test_find_best_split_one_instance() {
        let instances = vec![Instance {
            data: vec![1.0],
            label: 0,
        }];
        let result = find_best_split(&instances);
        assert!(result.is_none());
    }
    
    #[test]
    fn test_find_best_split_no_split() {
        let instances = vec![
            Instance {
                data: vec![1.0],
                label: 0,
            },
            Instance {
                data: vec![1.0],
                label: 0,
            },
        ];
        let result = find_best_split(&instances);
        assert!(result.is_none());
    }

    #[test]
    fn test_find_best_split_identical_features() {
        let instances = vec![
            Instance {
                data: vec![1.0],
                label: 0,
            },
            Instance {
                data: vec![1.0],
                label: 1,
            },
        ];
        let result = find_best_split(&instances);
        assert!(result.is_none());
    }

    #[test]
    fn test_gini_impurity_multiple_labels() {
        let instances = vec![
            Instance {
                data: vec![1.0],
                label: 0,
            },
            Instance {
                data: vec![2.0],
                label: 1,
            },
            Instance {
                data: vec![3.0],
                label: 2,
            },
            Instance {
                data: vec![4.0],
                label: 0,
            },
            Instance {
                data: vec![5.0],
                label: 1,
            },
            Instance {
                data: vec![6.0],
                label: 2,
            },
        ];
        let impurity = gini_impurity(&instances);
        // Expected Gini impurity: 1 - ( (2/6)^2 + (2/6)^2 + (2/6)^2 ) = 0.6666...
        assert!((impurity - 0.6666667).abs() < 1e-6);
    }
}
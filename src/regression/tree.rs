use std::iter::Sum;
use num_traits::{FromPrimitive, Num, ToPrimitive};


#[derive(Debug, Clone)]
pub(crate) struct InstanceRegression<D> {
    pub(crate) data: Vec<D>,
    pub(crate) target: D,
}

pub(crate) struct NodeRegression<D> {
    pub(crate) is_leaf: bool,
    pub(crate) prediction: Option<D>,
    pub(crate) feature_index: Option<usize>,
    pub(crate) threshold: Option<D>,
    pub(crate) left: Option<Box<NodeRegression<D>>>,
    pub(crate) right: Option<Box<NodeRegression<D>>>,
}

pub struct TreeRegression<D> {
    criterion: String,
    max_depth: usize,
    min_samples_split: usize,
    min_samples_leaf: usize,
    root: Option<Box<NodeRegression<D>>>,
    expected_feature_count: usize
}

impl<D> TreeRegression<D>
where
    D: Num + Copy + Clone + PartialOrd + ToPrimitive + FromPrimitive + Sum,
{
    pub fn new() -> Self {
        Self {
            criterion: "mae".to_string(),
            max_depth: usize::MAX,
            min_samples_split: 2,
            min_samples_leaf: 1,
            root: None,
            expected_feature_count: 0,
        }
    }


    /// Sets the criterion to be used for splitting nodes in the decision tree.
    ///
    /// # Arguments
    /// * `criterion` - A string slice that sets the splitting criterion (e.g., "mse", "huber").
    ///
    /// # Example
    /// ```
    /// use rusty_science::regression::TreeRegression;
    ///
    /// let mut classifier: TreeRegression<f64> = TreeRegression::new();
    /// classifier.set_criterion("mse");
    /// ```
    pub fn set_criterion(&mut self, criterion: &str) {
        self.criterion = criterion.to_string();
    }

    /// Sets the maximum depth of the decision tree.
    ///
    /// # Arguments
    /// * `max_depth` - The maximum depth that the tree is allowed to grow to.
    ///
    /// # Example
    /// ```
    /// use rusty_science::regression::TreeRegression;
    ///
    /// let mut classifier: TreeRegression<f64> = TreeRegression::new();
    /// classifier.set_max_depth(5);
    /// ```
    pub fn set_max_depth(&mut self, max_depth: usize) {
        self.max_depth = max_depth;
    }

    /// Sets the minimum number of samples required to split an internal node.
    ///
    /// # Arguments
    /// * `min_samples_split` - The minimum number of samples needed to attempt a split.
    ///
    /// # Example
    /// ```
    /// use rusty_science::regression::TreeRegression;
    ///
    /// let mut classifier: TreeRegression<f64> = TreeRegression::new();
    /// classifier.set_min_samples_split(4);
    /// ```
    pub fn set_min_samples_split(&mut self, min_samples_split: usize) {
        self.min_samples_split = min_samples_split;
    }

    /// Sets the minimum number of samples required to be in a leaf node.
    ///
    /// # Arguments
    /// * `min_samples_leaf` - The minimum number of samples a leaf must have.
    ///
    /// # Example
    /// ```
    /// use rusty_science::regression::TreeRegression;
    ///
    /// let mut classifier: TreeRegression<f64> = TreeRegression::new();
    /// classifier.set_min_samples_leaf(3);
    /// ```
    pub fn set_min_samples_leaf(&mut self, min_samples_leaf: usize) {
        self.min_samples_leaf = min_samples_leaf;
    }

    /// Trains the decision tree classifier on the provided dataset.
    ///
    /// # Arguments
    /// * `data` - A vector of vectors containing the feature data for training.
    /// * `label` - A vector containing the corresponding labels for each data sample.
    ///
    /// # Panics
    /// This function will panic if the provided dataset is empty.
    ///
    /// # Example
    /// ```
    /// use rusty_science::regression::TreeRegression;
    ///
    /// let data = vec![
    ///     vec![2.771244718, 1.784783929],
    ///     vec![1.728571309, 1.169761413],
    ///     vec![3.678319846, 2.81281357],
    /// ];
    /// let labels = vec![0.0, 0.0, 1.0];
    ///
    /// let mut classifier: TreeRegression<f64> = TreeRegression::new();
    /// classifier.fit(data, labels);
    /// ```
    pub fn fit(&mut self, data: Vec<Vec<D>>, target: Vec<D>) {
        if data.is_empty() {
            panic!("Training data cannot be empty.");
        }
        self._fit(data, target);
    }

    fn _fit(&mut self, data: Vec<Vec<D>>, targets: Vec<D>){
        self.expected_feature_count = data[0].len(); // Set expected feature count
        let instances: Vec<InstanceRegression<D>> = data
            .into_iter()
            .zip(targets.into_iter())
            .map(|(d, t)| InstanceRegression { data: d, target: t })
            .collect();
        self.root = Some(Box::from(self._build_tree(
            instances,
            0
        )));
    }

    fn _build_tree(&self, instances: Vec<InstanceRegression<D>>, depth: usize) -> NodeRegression<D> {
        if instances.is_empty() {
            panic!("No instances to split on.");
        }
        // Check if all labels are the same or max depth reached
        if depth >= self.max_depth || instances.len() < self.min_samples_split {
            // Create a leaf node with the most common label
            let prediction = self._mean_target(&instances);
            return NodeRegression {
                is_leaf: true,
                prediction: Some(prediction),
                feature_index: None,
                threshold: None,
                left: None,
                right: None,
            };
        }

        // Find the best split
        if let Some((best_feature, best_threshold, left_instances, right_instances)) =
            find_best_split_regression(&instances, &*self.criterion)
        {
            // Check for minimum samples in leaves
            if left_instances.len() < self.min_samples_leaf || right_instances.len() < self.min_samples_leaf {
                // Create a leaf node with the most common label
                let prediction = self._mean_target(&instances);
                return NodeRegression {
                    is_leaf: true,
                    prediction: Some(prediction),
                    feature_index: None,
                    threshold: None,
                    left: None,
                    right: None,
                };
            }

            // Recursively build the left and right subtrees
            let left_node = self._build_tree(left_instances, depth + 1);
            let right_node = self._build_tree(right_instances, depth + 1);

            NodeRegression {
                is_leaf: false,
                prediction: None,
                feature_index: Some(best_feature),
                threshold: best_threshold,
                left: Some(Box::new(left_node)),
                right: Some(Box::new(right_node)),
            }
        } else {
            // Cannot find a valid split, create a leaf node
            let prediction = self._mean_target(&instances);
            NodeRegression {
                is_leaf: true,
                prediction: Some(prediction),
                feature_index: None,
                threshold: None,
                left: None,
                right: None,
            }
        }
    }

    /// Predicts the label for a given input vector using the trained decision tree.
    ///
    /// # Arguments
    /// * `target` - A vector containing the features of the input sample to predict.
    ///
    /// # Returns
    /// The predicted label for the input sample.
    ///
    /// # Panics
    /// This function will panic if the decision tree has not been trained yet.
    ///
    /// # Example
    /// ```
    /// use rusty_science::regression::TreeRegression;
    ///
    /// let data = vec![
    ///     vec![2.771244718, 1.784783929],
    ///     vec![1.728571309, 1.169761413],
    ///     vec![3.678319846, 2.81281357],
    /// ];
    /// let labels = vec![0.0, 0.0, 1.0];
    ///
    /// let mut classifier: TreeRegression<f64> = TreeRegression::new();
    /// classifier.fit(data, labels);
    ///
    /// let test_sample = vec![3.0, 1.5];
    /// let prediction = classifier.predict(test_sample);
    /// ```
    pub fn predict(&self, target: Vec<D>) -> D {
        self._predict(self.root.as_deref(), &target)
    }

    fn _predict(&self, node: Option<&NodeRegression<D>>, target: &Vec<D>) -> D {
        if target.len() != self.expected_feature_count {
            panic!(
                "Input feature vector length ({}) does not match expected length ({})",
                target.len(),
                self.expected_feature_count
            );
        }
        match node {
            Some(n) => {
                if n.is_leaf {
                    n.prediction.unwrap()
                } else {
                    let feature_index = n.feature_index.unwrap();
                    let threshold = n.threshold.unwrap();
                    if feature_index >= target.len() {
                        panic!(
                            "Feature index {} out of bounds for input with length {}",
                            feature_index,
                            target.len()
                        );
                    }
                    if target[feature_index] <= threshold {
                        self._predict(n.left.as_deref(), target)
                    } else {
                        self._predict(n.right.as_deref(), target)
                    }
                }
            }
            None => panic!("The tree has not been trained."),
        }
    }
    
    fn _mean_target(&self, instances: &[InstanceRegression<D>]) -> D {
        let sum: D = instances.iter().map(|inst| inst.target).sum();
        let count = D::from_usize(instances.len()).unwrap();
        sum / count
    }
}

fn find_best_split_regression<D>(
    instances: &[InstanceRegression<D>], criterion: &str) -> Option<(usize, Option<D>, Vec<InstanceRegression<D>>, Vec<InstanceRegression<D>>)>
where
    D: Num + Copy + Clone + PartialOrd + ToPrimitive,
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
            let (left, right): (Vec<InstanceRegression<D>>, Vec<InstanceRegression<D>>) = instances
                .iter()
                .cloned()
                .partition(|inst| inst.data[feature_index] <= threshold);

            if left.is_empty() || right.is_empty() {
                continue;
            }

            let impurity_left = calculate_impurity(&left, criterion);
            let impurity_right = calculate_impurity(&right, criterion);

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

fn calculate_impurity<D>(instances: &[InstanceRegression<D>], criterion: &str) -> f64
where
    D: Num + Copy + Clone + ToPrimitive,
{
    match criterion {
        "mae" => mae_tree(instances),
        "mse" => mse_tree(instances),
        "huber" => huber_tree(instances, 1.0), // Assuming delta = 1.0 for Huber; you can make it adjustable
        _ => panic!("Unknown criterion: {}", criterion),
    }
}

fn mae_tree<D>(instances: &[InstanceRegression<D>]) -> f64
where
    D: Num + Copy + Clone + ToPrimitive,
{
    if instances.is_empty() {
        return 0.0;
    }

    let mean_target = instances
        .iter()
        .map(|inst| inst.target.to_f64().unwrap())
        .sum::<f64>()
        / instances.len() as f64;

    instances
        .iter()
        .map(|inst| (inst.target.to_f64().unwrap() - mean_target).abs())
        .sum::<f64>()
        / instances.len() as f64
}

fn mse_tree<D>(instances: &[InstanceRegression<D>]) -> f64
where
    D: Num + Copy + ToPrimitive,
{
    if instances.is_empty() {
        return 0.0;
    }
    let mean_target: f64 = instances
        .iter()
        .map(|inst| inst.target.to_f64().unwrap())
        .sum::<f64>()
        / instances.len() as f64;

    instances
        .iter()
        .map(|inst| {
            let error = inst.target.to_f64().unwrap() - mean_target;
            error * error
        })
        .sum::<f64>()
        / instances.len() as f64
}

fn huber_tree<D>(instances: &[InstanceRegression<D>], delta: f64) -> f64
where
    D: Num + Copy + Clone + ToPrimitive,
{
    if instances.is_empty() {
        return 0.0;
    }

    let mean_target = instances
        .iter()
        .map(|inst| inst.target.to_f64().unwrap())
        .sum::<f64>()
        / instances.len() as f64;

    instances
        .iter()
        .map(|inst| {
            let error = inst.target.to_f64().unwrap() - mean_target;
            if error.abs() <= delta {
                0.5 * error * error
            } else {
                delta * (error.abs() - 0.5 * delta)
            }
        })
        .sum::<f64>()
        / instances.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tree_repressor_initialization() {
        let tree: TreeRegression<f64> = TreeRegression::new();
        assert_eq!(tree.criterion, "mae");
        assert_eq!(tree.max_depth, usize::MAX);
        assert_eq!(tree.min_samples_split, 2);
        assert_eq!(tree.min_samples_leaf, 1);
        assert_eq!(tree.root.is_none(), true);
    }

    #[test]
    fn test_set_max_depth() {
        let mut tree: TreeRegression<f64> = TreeRegression::new();
        tree.set_max_depth(10);
        assert_eq!(tree.max_depth, 10);
    }

    #[test]
    fn test_set_min_samples_split() {
        let mut tree: TreeRegression<f64> = TreeRegression::new();
        tree.set_min_samples_split(5);
        assert_eq!(tree.min_samples_split, 5);
    }

    #[test]
    fn test_set_min_samples_leaf() {
        let mut tree: TreeRegression<f64> = TreeRegression::new();
        tree.set_min_samples_leaf(4);
        assert_eq!(tree.min_samples_leaf, 4);
    }

    #[test]
    fn test_fit_and_predict_exact_match() {
        // Simple dataset where the prediction should be exactly the mean of targets
        let data = vec![
            vec![1.0, 2.0],
            vec![1.0, 2.0],
            vec![1.0, 2.0],
        ];
        let targets = vec![10.0, 10.0, 10.0];

        let mut tree: TreeRegression<f64> = TreeRegression::new();
        tree.fit(data.clone(), targets.clone());

        // Predicting on the same data points should yield the exact target value
        let prediction = tree.predict(vec![1.0, 2.0]);
        assert_eq!(prediction, 10.0);
    }

    #[test]
    fn test_fit_and_predict_average_target() {
        // Dataset designed to yield a mean target as the output in a leaf node
        let data = vec![
            vec![1.0, 2.0],
            vec![1.0, 2.0],
            vec![1.0, 2.0],
        ];
        let targets = vec![10.0, 20.0, 30.0];

        let mut tree: TreeRegression<f64> = TreeRegression::new();
        tree.fit(data.clone(), targets.clone());

        // The prediction should be the average of the targets: (10 + 20 + 30) / 3 = 20.0
        let prediction = tree.predict(vec![1.0, 2.0]);
        assert!((prediction - 20.0).abs() < 1e-6);
    }

    #[test]
    fn test_predict_split_node() {
        // A dataset that can be split to test if predictions vary correctly
        let data = vec![
            vec![1.0, 2.0],
            vec![1.0, 3.0],
            vec![2.0, 2.0],
            vec![2.0, 3.0],
        ];
        let targets = vec![10.0, 20.0, 30.0, 40.0];

        let mut tree: TreeRegression<f64> = TreeRegression::new();
        tree.set_max_depth(1); 
        tree.fit(data.clone(), targets.clone());

        let prediction1 = tree.predict(vec![1.0, 2.0]);
        let prediction2 = tree.predict(vec![2.0, 3.0]);

        assert!((prediction1 - 15.0).abs() < 1e-6); // Mean of [10.0, 20.0]
        assert!((prediction2 - 35.0).abs() < 1e-6); // Mean of [30.0, 40.0]
    }

    #[test]
    #[should_panic(expected = "Training data cannot be empty.")]
    fn test_fit_empty_data() {
        let data: Vec<Vec<f64>> = vec![];
        let target: Vec<f64> = vec![];

        let mut tree: TreeRegression<f64> = TreeRegression::new();
        tree.fit(data, target);
    }

    #[test]
    fn test_predict_with_invalid_length() {
        let data = vec![
            vec![1.0, 2.0],
            vec![2.0, 3.0]
        ];
        let targets = vec![5.0, 10.0];

        let mut tree: TreeRegression<f64> = TreeRegression::new();
        tree.fit(data, targets);

        let prediction = std::panic::catch_unwind(|| {
            tree.predict(vec![1.0, 2.0, 3.0]) // Invalid length
        });
        assert!(prediction.is_err());
    }

    #[test]
    fn test_find_best_split_regression() {
        let instances = vec![
            InstanceRegression { data: vec![1.0, 2.0], target: 5.0 },
            InstanceRegression { data: vec![2.0, 3.0], target: 10.0 },
            InstanceRegression { data: vec![3.0, 4.0], target: 15.0 }
        ];

        let split = find_best_split_regression(&instances, "mse");
        assert!(split.is_some());
    }

    #[test]
    fn test_mae_tree() {
        let instances = vec![
            InstanceRegression { data: vec![1.0, 2.0], target: 5.0 },
            InstanceRegression { data: vec![2.0, 3.0], target: 7.0 },
            InstanceRegression { data: vec![3.0, 4.0], target: 6.0 }
        ];

        let mae = mae_tree(&instances);
        assert!(mae >= 0.0); 
    }
}


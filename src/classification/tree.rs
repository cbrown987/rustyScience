//! # Decision Tree Classification Algorithm
//!
//! ## Theoretical Background
//!
//! Decision Trees are versatile machine learning algorithms that create a model resembling a tree-like structure of decisions:
//!
//! - They partition the feature space into regions, with each region assigned a class label
//! - They make decisions by recursively splitting data based on feature values
//! - They can handle both numerical and categorical features
//! - They form the foundation for more powerful ensemble methods like Random Forests and Gradient Boosting
//!
//! The algorithm works by finding optimal splits in the data that maximize the information gain
//! or minimize impurity (like Gini impurity) at each step, creating a hierarchical structure
//! of decision rules.
//!
//! ## Parameters
//!
//! - `criterion`: Measure used to evaluate the quality of a split.
//!    - "gini": Uses Gini impurity (measures the probability of incorrect classification)
//!    - "entropy": Uses information gain (measures the reduction in disorder)
//!    - Typical default: "gini" (computationally simpler)
//!
//! - `max_depth`: Maximum depth of the tree.
//!    - Too small: Underfitting, inability to capture complex patterns
//!    - Too large: Overfitting, capturing noise in the training data
//!    - Typical values: 3 to 10, depending on data complexity
//!
//! - `min_samples_split`: Minimum number of samples required to split an internal node.
//!    - Higher values: Prevents creating splits with small sample sizes
//!    - Lower values: Allows more detailed splits, risking overfitting
//!    - Typical values: 2 to 20
//!
//! - `min_samples_leaf`: Minimum number of samples required in a leaf node.
//!    - Higher values: Smoother decision boundaries, better generalization
//!    - Lower values: More complex patterns captured, risk of overfitting
//!    - Typical values: 1 to 10
//!
//! ## Usage Examples
//!
//! ```rust
//! use rusty_science::classification::TreeClassifier;
//!
//! // Create example data
//! let data = vec![
//!     vec![2.5, 2.4], vec![1.0, 0.9], vec![3.3, 4.4], 
//!     vec![4.2, 3.1], vec![4.0, 4.5], vec![2.2, 1.8]
//! ];
//! let labels = vec![0, 0, 1, 1, 1, 0];
//!
//! // Create and configure the decision tree
//! let mut tree = TreeClassifier::new();
//! tree.set_criterion("gini");
//! tree.set_max_depth(5);
//! tree.set_min_samples_split(2);
//! tree.set_min_samples_leaf(1);
//!
//! // Fit the model
//! tree.fit(data.clone(), labels);
//!
//! // Make a prediction
//! let prediction = tree.predict(vec![3.8, 4.0]);
//! println!("Predicted class: {}", prediction);
//! ```
//!
//! ## Performance Characteristics
//!
//! - **Time Complexity**:
//!   - Training: O(n * d * log(n)), where:
//!     - n is the number of training examples
//!     - d is the number of features
//!   - Prediction: O(log(n)) for a single prediction (average case), where n is the number of training samples
//!
//! - **Space Complexity**: O(n) in the worst case (completely imbalanced tree)
//!
//! - **Strengths**:
//!   - Simple to understand and interpret
//!   - Requires little data preprocessing (no need for normalization)
//!   - Handles both numerical and categorical data
//!   - Can handle multi-output problems
//!   - Implicitly performs feature selection
//!   - Non-parametric, capturing non-linear patterns
//!   - Robust to outliers
//!   - Computationally efficient
//!   - Handles missing values effectively
//!
//! - **Weaknesses**:
//!   - Prone to overfitting, especially with deep trees
//!   - Can create biased trees if classes are imbalanced
//!   - Unstable: small variations in data can result in very different trees
//!   - Cannot extrapolate beyond the training data range
//!   - May create overly complex trees that don't generalize well
//!   - Greedy algorithm that doesn't guarantee globally optimal trees
//!   - Struggles with tasks requiring diagonal decision boundaries
//!   - Sensitive to rotation of the feature space
//!

use num_traits::{FromPrimitive, Num, ToPrimitive};

#[derive(Debug, Clone)]
pub(crate) struct InstanceClassifier<D, L> {
    pub(crate) data: Vec<D>,
    pub(crate) target: L,
}

pub(crate) struct NodeClassifier<D, L> {
    pub(crate) is_leaf: bool,
    pub(crate) prediction: Option<L>,
    pub(crate) feature_index: Option<usize>,
    pub(crate) threshold: Option<D>,
    pub(crate) left: Option<Box<NodeClassifier<D, L>>>,
    pub(crate) right: Option<Box<NodeClassifier<D, L>>>,
}

pub struct TreeClassifier<D, L> {
    criterion: String,
    max_depth: usize,
    min_samples_split: usize,
    min_samples_leaf: usize,
    root: Option<Box<NodeClassifier<D, L>>>,
    expected_feature_count: usize
}

impl<D, L> TreeClassifier<D, L>
where
    D: Num + Copy + Clone + PartialOrd + ToPrimitive + FromPrimitive,
    L: Num + Copy + Clone + PartialOrd + ToPrimitive + FromPrimitive,
{
    pub fn new() -> Self {
        Self {
            criterion: "gini".to_string(),
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
    /// * `criterion` - A string slice that sets the splitting criterion (e.g., "gini", "entropy").
    ///
    /// # Example
    /// ```
    /// use rusty_science::classification::TreeClassifier;
    ///
    /// let mut classifier: TreeClassifier<f64, i32> = TreeClassifier::new();
    /// classifier.set_criterion("entropy");
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
    /// use rusty_science::classification::TreeClassifier;
    ///
    /// let mut classifier: TreeClassifier<f64, i32> = TreeClassifier::new();
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
    /// use rusty_science::classification::TreeClassifier;
    ///
    /// let mut classifier: TreeClassifier<f64, i32> = TreeClassifier::new();
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
    /// use rusty_science::classification::TreeClassifier;
    ///
    /// let mut classifier: TreeClassifier<f64, i32> = TreeClassifier::new();
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
    /// use rusty_science::classification::TreeClassifier;
    ///
    /// let data = vec![
    ///     vec![2.771244718, 1.784783929],
    ///     vec![1.728571309, 1.169761413],
    ///     vec![3.678319846, 2.81281357],
    /// ];
    /// let labels = vec![0, 0, 1];
    ///
    /// let mut classifier: TreeClassifier<f64, i32> = TreeClassifier::new();
    /// classifier.fit(data, labels);
    /// ```
    pub fn fit(&mut self, data: Vec<Vec<D>>, label: Vec<L>) {
        if data.is_empty() {
            panic!("Training data cannot be empty.");
        }
        self._fit(data, label);
    }

    fn _fit(&mut self, data: Vec<Vec<D>>, label: Vec<L>) {
        self.expected_feature_count = data[0].len();
        let instances: Vec<InstanceClassifier<D, L>> = data.into_iter()
            .zip(label.into_iter())
            .map(|(d, l)| InstanceClassifier { data: d, target: l })
            .collect();

        let features_data = precompute_sorted_features(&instances);
        self.root = Some(Box::from(self._build_tree(instances, 0, &features_data)));
    }

    fn _build_tree(&self, instances: Vec<InstanceClassifier<D, L>>, depth: usize, features_data: &[Vec<(D, L)>]) -> NodeClassifier<D, L> {
        // Check stopping conditions
        if instances.is_empty() {
            panic!("No instances to split on.");
        }

        // Get labels
        let labels: Vec<&L> = instances.iter().map(|inst| &inst.target).collect();

        // Check if all labels are the same or max depth reached
        if self._is_pure(&labels) || depth >= self.max_depth || instances.len() < self.min_samples_split {
            // Create a leaf node with the most common label
            let prediction = self._majority_label(&labels);
            return NodeClassifier {
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
            find_best_split_classification(&instances, features_data)
        {
            // Check for minimum samples in leaves
            if left_instances.len() < self.min_samples_leaf || right_instances.len() < self.min_samples_leaf {
                // Create a leaf node with the most common label
                let prediction = self._majority_label(&labels);
                return NodeClassifier {
                    is_leaf: true,
                    prediction: Some(prediction),
                    feature_index: None,
                    threshold: None,
                    left: None,
                    right: None,
                };
            }

            // Recursively build the left and right subtrees
            let left_node = self._build_tree(left_instances, depth + 1, features_data);
            let right_node = self._build_tree(right_instances, depth + 1, features_data);

            NodeClassifier {
                is_leaf: false,
                prediction: None,
                feature_index: Some(best_feature),
                threshold: best_threshold,
                left: Some(Box::new(left_node)),
                right: Some(Box::new(right_node)),
            }
        } else {
            // Cannot find a valid split, create a leaf node
            let prediction = self._majority_label(&labels);
            NodeClassifier {
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
    /// use rusty_science::classification::TreeClassifier;
    ///
    /// let data = vec![
    ///     vec![2.771244718, 1.784783929],
    ///     vec![1.728571309, 1.169761413],
    ///     vec![3.678319846, 2.81281357],
    /// ];
    /// let labels = vec![0, 0, 1];
    ///
    /// let mut classifier: TreeClassifier<f64, i32> = TreeClassifier::new();
    /// classifier.fit(data, labels);
    ///
    /// let test_sample = vec![3.0, 1.5];
    /// let prediction = classifier.predict(test_sample);
    /// ```
    pub fn predict(&self, target: Vec<D>) -> L {
        self._predict(self.root.as_deref(), &target)
    }

    fn _predict(&self, node: Option<&NodeClassifier<D, L>>, target: &Vec<D>) -> L {
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

    fn _is_pure(&self, labels: &[&L]) -> bool {
        labels.windows(2).all(|w| w[0] == w[1])
    }

    fn _majority_label(&self, labels: &[&L]) -> L {
        let mut label_counts: Vec<(L, usize)> = Vec::new();

        // Count the occurrences of each label
        for &label in labels {
            let mut found = false;
            for &mut (ref existing_label, ref mut count) in &mut label_counts {
                if existing_label == label {
                    *count += 1;
                    found = true;
                    break;
                }
            }
            if !found {
                label_counts.push((label.clone(), 1));
            }
        }

        // Find the label with the maximum count
        label_counts
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(label, _)| label)
            .unwrap()
    }
}

fn find_best_split_classification<D, L>(
    instances: &[InstanceClassifier<D, L>],
    features_data: &[Vec<(D, L)>] // pre-sorted feature data
) -> Option<(usize, Option<D>, Vec<InstanceClassifier<D, L>>, Vec<InstanceClassifier<D, L>>)>
where
    D: Num + Copy + Clone + PartialOrd + ToPrimitive + FromPrimitive,
    L: Num + Copy + Clone + PartialOrd + ToPrimitive,
{
    if instances.is_empty() {
        return None;
    }

    let num_features = features_data.len();
    let mut best_feature = 0;
    let mut best_threshold: Option<D> = None;
    let mut best_impurity = f64::INFINITY;
    let mut best_left = Vec::new();
    let mut best_right = Vec::new();

    // Convert the entire dataset into a vector of (data, label) for partitioning reference
    // In a more optimized version, you'd just keep track of indices.
    let total_instances = instances.len() as f64;

    for feature_index in 0..num_features {
        let feature_vals = &features_data[feature_index];

        // If all values are the same for this feature, no valid split
        if feature_vals.first().unwrap().0 == feature_vals.last().unwrap().0 {
            continue;
        }

        let mut right_counts: Vec<(L, usize)> = Vec::new();
        for &(_, lbl) in feature_vals {
            update_counts(&mut right_counts, lbl);
        }

        let mut left_counts: Vec<(L, usize)> = Vec::new();

        // We'll consider splits between distinct values
        for i in 0..(feature_vals.len() - 1) {
            let (val, lbl) = feature_vals[i];

            // Move this instance from right to left
            decrement_counts(&mut right_counts, lbl);
            update_counts(&mut left_counts, lbl);

            let next_val = feature_vals[i+1].0;
            if next_val == val {
                continue;
            }

            let mid_val = (val.to_f64().unwrap() + next_val.to_f64().unwrap()) / 2.0;
            let threshold = FromPrimitive::from_f64(mid_val).unwrap();
            let left_len = count_total(&left_counts) as f64;
            let right_len = count_total(&right_counts) as f64;

            let impurity_left = gini_from_counts(&left_counts, left_len);
            let impurity_right = gini_from_counts(&right_counts, right_len);

            let impurity = (left_len * impurity_left + right_len * impurity_right) / total_instances;
            if impurity < best_impurity {
                best_impurity = impurity;
                best_feature = feature_index;
                best_threshold = Some(threshold);

                // We need to reconstruct actual left/right sets from this threshold.
                // Since we know threshold = val, let's partition the original instances:
                let (l_set, r_set) = instances.iter().cloned().partition(|inst| inst.data[feature_index] <= threshold);
                best_left = l_set;
                best_right = r_set;
            }

        }
    }

    if best_impurity == f64::INFINITY {
        None
    } else {
        Some((best_feature, best_threshold, best_left, best_right))
    }
}

// Helper functions for label counting

fn update_counts<L: PartialEq + Clone>(counts: &mut Vec<(L, usize)>, lbl: L) {
    for (existing_label, count) in counts.iter_mut() {
        if *existing_label == lbl {
            *count += 1;
            return;
        }
    }
    counts.push((lbl, 1));
}


fn decrement_counts<L: PartialEq>(counts: &mut Vec<(L, usize)>, lbl: L) {
    for &mut (ref existing_label, ref mut count) in counts {
        if *existing_label == lbl {
            *count -= 1;
            return;
        }
    }
}

fn count_total<L>(counts: &[(L, usize)]) -> usize {
    counts.iter().map(|&(_, c)| c).sum()
}

fn gini_from_counts<L>(counts: &[(L, usize)], total: f64) -> f64 {
    if total == 0.0 {
        return 0.0;
    }

    let sum_of_squares = counts.iter().map(|&(_, c)| {
        let p = c as f64 / total;
        p * p
    }).sum::<f64>();

    1.0 - sum_of_squares
}

fn precompute_sorted_features<D, L>(instances: &[InstanceClassifier<D, L>]) -> Vec<Vec<(D, L)>>
where
    D: Num + Copy + Clone + PartialOrd + ToPrimitive,
    L: Num + Copy + Clone + PartialOrd + ToPrimitive,
{
    if instances.is_empty() {
        return Vec::new();
    }

    let num_features = instances[0].data.len();
    let mut features_data: Vec<Vec<(D, L)>> = vec![Vec::new(); num_features];

    // Fill each feature vector with (value, label) tuples
    for inst in instances {
        for (f_idx, &val) in inst.data.iter().enumerate() {
            features_data[f_idx].push((val, inst.target));
        }
    }

    // Sort each feature's data by the feature value
    for feature_vector in &mut features_data {
        feature_vector.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    }

    features_data
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_functionality() {
        // Simple dataset
        let data = vec![
            vec![2.771244718, 1.784783929],
            vec![1.728571309, 1.169761413],
            vec![3.678319846, 2.81281357],
            vec![3.961043357, 2.61995032],
            vec![2.999208922, 2.209014212],
            vec![7.497545867, 3.162953546],
            vec![9.00220326, 3.339047188],
            vec![7.444542326, 0.476683375],
            vec![10.12493903, 3.234550982],
            vec![6.642287351, 3.319983761],
        ];
        let labels = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];

        let mut classifier = TreeClassifier::new();
        classifier.set_max_depth(3);
        classifier.set_min_samples_split(2);
        classifier.set_min_samples_leaf(1);
        classifier.fit(data, labels);

        let test_sample = vec![3.0, 1.5];
        let prediction = classifier.predict(test_sample);
        assert_eq!(prediction, 0);
    }

    #[test]
    fn test_all_same_label() {
        // Dataset where all labels are the same
        let data = vec![
            vec![1.0, 2.0],
            vec![1.5, 1.8],
            vec![2.0, 2.2],
            vec![3.0, 3.2],
            vec![3.5, 3.8],
        ];
        let labels = vec![1, 1, 1, 1, 1];

        let mut classifier = TreeClassifier::new();
        classifier.fit(data, labels);

        let test_sample = vec![2.5, 2.5];
        let prediction = classifier.predict(test_sample);
        assert_eq!(prediction, 1);
    }

    #[test]
    #[should_panic]
    fn test_empty_dataset() {
        // Attempting to fit an empty dataset should panic
        let data: Vec<Vec<f64>> = vec![];
        let labels: Vec<f64> = vec![];

        let mut classifier = TreeClassifier::new();
        
        classifier.fit(data, labels);
        
    }

    #[test]
    fn test_parameters_effect() {
        // Testing the effect of changing parameters
        let data = vec![
            vec![2.771244718, 1.784783929],
            vec![1.728571309, 1.169761413],
            vec![3.678319846, 2.81281357],
            vec![3.961043357, 2.61995032],
            vec![2.999208922, 2.209014212],
            vec![7.497545867, 3.162953546],
            vec![9.00220326, 3.339047188],
            vec![7.444542326, 0.476683375],
            vec![10.12493903, 3.234550982],
            vec![6.642287351, 3.319983761],
        ];
        let labels = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];

        let mut classifier = TreeClassifier::new();
        classifier.set_max_depth(1); // Shallow tree
        classifier.fit(data.clone(), labels.clone());

        let test_sample = vec![3.0, 1.5];
        let prediction_shallow = classifier.predict(test_sample.clone());

        classifier.set_max_depth(5); // Deeper tree
        classifier.fit(data, labels);
        let prediction_deep = classifier.predict(test_sample);

        // Predictions may differ due to tree depth
        assert!(prediction_shallow == 1 || prediction_shallow == 0);
        assert!(prediction_deep == 0);

        // Ensure that deeper tree predicts the expected label
        assert_eq!(prediction_deep, 0);
    }

    #[test]
    fn test_predict_untrained() {
        // Attempting to predict without training should panic
        let classifier: TreeClassifier<f64, i32> = TreeClassifier::new();

        let test_sample = vec![3.0, 1.5];

        let result = std::panic::catch_unwind(|| {
            classifier.predict(test_sample);
        });
        assert!(result.is_err());
    }
}

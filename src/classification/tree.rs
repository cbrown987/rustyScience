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
    D: Num + Copy + Clone + PartialOrd + ToPrimitive,
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
    
    fn _fit(&mut self, data: Vec<Vec<D>>, label: Vec<L>){
        self.expected_feature_count = data[0].len(); // Set expected feature count
        let instances: Vec<InstanceClassifier<D, L>> = data
            .into_iter()
            .zip(label.into_iter())
            .map(|(d, l)| InstanceClassifier { data: d, target: l })
            .collect();
        self.root = Some(Box::from(self._build_tree(
            instances,
            0, // current depth
        )));
    }

    fn _build_tree(&self, instances: Vec<InstanceClassifier<D, L>>, depth: usize) -> NodeClassifier<D, L> {
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
            find_best_split_classification(&instances)
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
            let left_node = self._build_tree(left_instances, depth + 1);
            let right_node = self._build_tree(right_instances, depth + 1);

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
    instances: &[InstanceClassifier<D, L>]) -> Option<(usize, Option<D>, Vec<InstanceClassifier<D, L>>, Vec<InstanceClassifier<D, L>>)>
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
            let (left, right): (Vec<InstanceClassifier<D, L>>, Vec<InstanceClassifier<D, L>>) = instances
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

pub(crate) fn gini_impurity<D, L>(instances: &[InstanceClassifier<D, L>]) -> f64
where
    L: PartialEq + Clone,
{
    let mut label_counts: Vec<(L, usize)> = Vec::new();

    for instance in instances {
        let mut found = false;
        for (label, count) in &mut label_counts {
            if *label == instance.target {
                *count += 1;
                found = true;
                break;
            }
        }
        if !found {
            label_counts.push((instance.target.clone(), 1));
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

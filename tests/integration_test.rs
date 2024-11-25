mod test_metrics;
mod test_knn;
mod test_tree;
mod test_svc;

#[cfg(test)]
mod tests {
    // Knn
    use crate::test_knn::test_knn_classification::test_knn_classification_integration;
    use crate::test_knn::test_knn_clustering::test_knn_clustering_integration;
    use crate::test_knn::test_knn_regression::test_knn_regression_integration;

    
    // metrics
    use crate::test_metrics::accuracy_score::{test_accuracy_score, test_accuracy_score_normalized};
    use crate::test_metrics::r2::test_all_r2;

    #[test]
    fn test_knn(){
        test_knn_regression_integration();
        test_knn_classification_integration();
        test_knn_clustering_integration();
    }
    
    #[test]
    fn test_metrics(){
        test_accuracy_score();
        test_accuracy_score_normalized();
        test_all_r2()
    }
    
    
}



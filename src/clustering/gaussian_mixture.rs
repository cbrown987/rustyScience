//! # Gaussian Mixture Model (GMM)
//!
//! A Gaussian Mixture Model is a probabilistic model that assumes data points are 
//! generated from a mixture of several Gaussian distributions with unknown parameters.
//! GMMs are used for clustering, density estimation, and generating synthetic samples.

use crate::common::utils::array2_to_vec;
use ndarray::{s, Array1, Array2, ArrayView1, Axis};
use num_traits::{Float, One, ToPrimitive, Zero};
use rand::prelude::*;
use rand::Rng;
use std::fmt::Debug;

/// Gaussian Mixture Model for clustering and density estimation
pub struct GaussianMixture<D, L> {
    data: Vec<Vec<D>>,
    labels: Vec<L>,
    n_components: usize,
    max_iter: usize,
    tol: f64,
    init_method: String,
    covariance_type: String,
    means: Option<Array2<f64>>,
    covariances: Option<Array2<f64>>,
    weights: Option<Array1<f64>>,
}

impl<D, L> GaussianMixture<D, L>
where
    D: Clone + Copy + PartialOrd + Float + Zero + One + Debug + ToPrimitive,
    L: Clone + Copy + Debug,
{
    /// Create a new Gaussian Mixture Model with default parameters
    pub fn new() -> Self {
        GaussianMixture {
            data: vec![],
            labels: vec![],
            n_components: 2,
            max_iter: 100,
            tol: 1e-3,
            init_method: "kmeans".to_string(),
            covariance_type: "full".to_string(),
            means: None,
            covariances: None,
            weights: None,
        }
    }

    /// Set the number of Gaussian components in the mixture
    pub fn set_n_components(&mut self, n_components: usize) {
        assert!(n_components > 0, "Number of components must be positive");
        self.n_components = n_components;
    }

    /// Set the maximum number of EM iterations
    pub fn set_max_iter(&mut self, max_iter: usize) {
        self.max_iter = max_iter;
    }

    /// Set the convergence tolerance
    pub fn set_tol(&mut self, tol: f64) {
        assert!(tol > 0.0, "Tolerance must be positive");
        self.tol = tol;
    }

    /// Set the initialization method: 'kmeans' or 'random'
    pub fn set_init_method(&mut self, init_method: &str) {
        assert!(
            init_method == "kmeans" || init_method == "random",
            "Init method must be 'kmeans' or 'random'"
        );
        self.init_method = init_method.to_string();
    }

    /// Set the covariance type: 'full', 'tied', 'diag', or 'spherical'
    pub fn set_covariance_type(&mut self, covariance_type: &str) {
        assert!(
            covariance_type == "full" || covariance_type == "tied" ||
                covariance_type == "diagonal" || covariance_type == "spherical",
            "Covariance type must be 'full', 'tied', 'diagonal', or 'spherical'"
        );
        self.covariance_type = covariance_type.to_string();
    }

    /// Fit the Gaussian Mixture Model to the training data
    pub fn fit(&mut self, data: Vec<Vec<D>>, labels: Vec<L>) {
        // Convert data to Array2<f64> for easier manipulation
        let array_data = self.convert_to_array2f64(&data);
        let n_samples = array_data.nrows();
        let n_features = array_data.ncols();
        
        // Initialize parameters (means, covariances, weights)
        self.initialize_parameters(&array_data);
        
        // Iterations for EM algorithm
        let mut log_likelihood_prev = f64::NEG_INFINITY;
        
        for _ in 0..self.max_iter {
            // E-step: Calculate responsibilities
            let (responsibilities, log_likelihood) = self.e_step(&array_data);
            
            // Check for convergence
            if (log_likelihood - log_likelihood_prev).abs() < self.tol {
                break;
            }
            log_likelihood_prev = log_likelihood;
            
            // M-step: Update parameters
            self.m_step(&array_data, &responsibilities);
        }
    }

    /// Get the cluster assignments for each data point
    pub fn predict(&self, data: Vec<Vec<D>>) -> Vec<usize> {
        if self.means.is_none() || self.covariances.is_none() || self.weights.is_none() {
            panic!("Model must be fit before prediction");
        }

        let n_samples = data.len();

        let data_array = self.convert_to_array2f64(&data);

        let means = self.means.as_ref().unwrap();
        let covariances = self.covariances.as_ref().unwrap();
        let weights = self.weights.as_ref().unwrap();

        let mut predictions = Vec::with_capacity(n_samples);

        // For each sample, compute the log probability for each component
        // and choose the component with the highest probability
        for i in 0..n_samples {
            let x_i = data_array.row(i).to_owned();
            let mut max_log_prob = f64::NEG_INFINITY;
            let mut max_component = 0;

            for k in 0..self.n_components {
                let weight_k = weights[k];
                let mean_k = means.row(k).to_owned();

                let cov_k = self.get_covariance_for_component(k, &mean_k, covariances);

                // Compute log probability
                let log_prob = self.multivariate_normal_pdf(&x_i, &mean_k, &cov_k).ln() + weight_k.ln();

                if log_prob > max_log_prob {
                    max_log_prob = log_prob;
                    max_component = k;
                }
            }

            predictions.push(max_component);
        }

        predictions
    }

    /// Compute the probability of each sample belonging to each Gaussian component
    pub fn predict_probability(&self, data: Vec<Vec<D>>) -> Array2<f64> {
        if self.means.is_none() || self.covariances.is_none() || self.weights.is_none() {
            panic!("Model not fitted yet. Call fit() before predict_probability().");
        }
        
        let array_data = self.convert_to_array2f64(&data);
        let n_samples = array_data.nrows();
        let n_components = self.n_components;
        
        // Initialize the result array
        let mut probabilities = Array2::<f64>::zeros((n_samples, n_components));
        
        // For each sample and component, calculate the weighted probability
        for i in 0..n_samples {
            let x_i = array_data.row(i).to_owned();
            
            // Calculate unnormalized probabilities for this sample across all components
            let mut sample_probs = Vec::with_capacity(n_components);
            let mut prob_sum = 0.0;
            
            for k in 0..n_components {
                let mean_k = self.means.as_ref().unwrap().row(k).to_owned();
                let cov_k = self.get_covariance_for_component(k, &mean_k, 
                                                             self.covariances.as_ref().unwrap());
                let weight_k = self.weights.as_ref().unwrap()[k];
                
                // Calculate probability density for this component
                let prob = weight_k * self.multivariate_normal_pdf(&x_i, &mean_k, &cov_k);
                sample_probs.push(prob);
                prob_sum += prob;
            }
            
            // Normalize the probabilities so they sum to 1
            for (k, prob) in sample_probs.iter().enumerate() {
                probabilities[[i, k]] = prob / prob_sum;
            }
        }
        
        probabilities
    }

    /// Compute the log-likelihood of the data under the model
    pub fn score(&self, data: &[Vec<D>]) -> f64 {
        // Check if the model has been fit
        if self.means.is_none() || self.covariances.is_none() || self.weights.is_none() {
            panic!("Model must be fit before scoring");
        }

        // Convert data to ndarray format
        let n_samples = data.len();
        if n_samples == 0 {
            return 0.0; // Or handle empty data differently
        }

        let n_features = data[0].len();
        let data_array = self.convert_to_array2f64(&data);


        // Get the model parameters
        let means = self.means.as_ref().unwrap();
        let covariances = self.covariances.as_ref().unwrap();
        let weights = self.weights.as_ref().unwrap();

        let mut weighted_log_prob = Array2::<f64>::zeros((n_samples, self.n_components));

        // For each component, compute log probability for each sample
        for k in 0..self.n_components {
            let weight_k = weights[k];
            // Ensure weight is positive before taking log
            if weight_k <= 0.0 {
                continue; // Skip this component or handle appropriately
            }

            let mean_k = means.row(k);

            // Get appropriate covariance matrix once per component
            let cov_k = match self.covariance_type.as_str() {
                "full" => {
                    // Assuming covariances is properly structured for full covariance
                    let start = k * n_features;
                    let end = (k + 1) * n_features;
                    covariances.slice(s![start..end, ..]).to_owned()
                },
                "diagonal" => {
                    let diag_values = covariances.row(k).to_owned();
                    Array2::from_diag(&diag_values)
                },
                "spherical" => {
                    let sigma = covariances[[k, 0]];
                    Array2::eye(n_features) * sigma
                },
                _ => panic!("Unsupported covariance type"),
            };

            // Compute log probability for all samples for this component
            for i in 0..n_samples {
                let x_i = data_array.row(i);
                // Compute log probability
                let log_prob = self.multivariate_normal_pdf(&x_i.to_owned(), &mean_k.to_owned(), &cov_k).ln();
                weighted_log_prob[[i, k]] = log_prob + weight_k.ln();
            }
        }

        // Compute log sum exp (to avoid underflow)
        let mut log_likelihood = 0.0;
        for i in 0..n_samples {
            let row_slice = weighted_log_prob.slice(s![i, ..]);
            // Find max for numerical stability
            let max_value = row_slice.fold(f64::NEG_INFINITY, |a, &b| a.max(b));

            // Handle extreme cases
            if max_value == f64::NEG_INFINITY {
                continue; // All probabilities are -inf, skip this sample or handle appropriately
            }

            // Compute log-sum-exp
            let sum_exp: f64 = row_slice
                .iter()
                .map(|&x| (x - max_value).exp())
                .sum();

            // Avoid log of zero
            if sum_exp > 0.0 {
                log_likelihood += max_value + sum_exp.ln();
            } else {
                log_likelihood += max_value; // Just use the max if sum is zero
            }
        }

        log_likelihood
    }

    /// Sample new data points from the fitted mixture model
    pub fn sample(&self, n_samples: usize) -> Vec<Vec<f64>> {
        let mut rng = thread_rng();
        let mut samples = Vec::with_capacity(n_samples);

        // Extract means, covariances, and weights from the model
        let means = self.means.as_ref().expect("Model not fitted");
        let covs = self.covariances.as_ref().expect("Model not fitted");
        let weights = self.weights.as_ref().expect("Model not fitted");

        // Convert ndarray means to Vec<Vec<f64>>
        let means_vec: Vec<Vec<f64>> = (0..means.shape()[0])
            .map(|i| (0..means.shape()[1])
                .map(|j| means[[i, j]])
                .collect())
            .collect();

        // Convert ndarray covariances to Vec<Vec<Vec<f64>>>
        let n_components = self.n_components;
        let n_features = means_vec[0].len();

        let covs_vec: Vec<Vec<Vec<f64>>> = match self.covariance_type.as_str() {
            "spherical" => {
                (0..n_components)
                    .map(|i| {
                        // Create diagonal matrix with the single variance value
                        let sigma = covs[[i, 0]];
                        (0..n_features)
                            .map(|j| {
                                (0..n_features)
                                    .map(|k| if j == k { sigma } else { 0.0 })
                                    .collect()
                            })
                            .collect()
                    })
                    .collect()
            },
            "diagonal" => {
                (0..n_components)
                    .map(|i| {
                        // Create diagonal matrix with diagonal variances
                        (0..n_features)
                            .map(|j| {
                                (0..n_features)
                                    .map(|k| if j == k { covs[[i, j]] } else { 0.0 })
                                    .collect()
                            })
                            .collect()
                    })
                    .collect()
            },
            "full" => {
                // Assumes covariance matrices are stored contiguously
                (0..n_components)
                    .map(|i| {
                        (0..n_features)
                            .map(|j| {
                                (0..n_features)
                                    .map(|k| covs[[i * n_features + j, k]])
                                    .collect()
                            })
                            .collect()
                    })
                    .collect()
            },
            _ => panic!("Unsupported covariance type"),
        };

        // Convert weights to Vec<f64>
        let weights_vec: Vec<f64> = weights.iter().copied().collect();

        // Generate samples
        for _ in 0..n_samples {
            // Choose component according to weights
            let u: f64 = rng.gen();
            let mut cumsum = 0.0;
            let mut selected_component = 0;

            for (i, &weight) in weights_vec.iter().enumerate() {
                cumsum += weight;
                if u <= cumsum {
                    selected_component = i;
                    break;
                }
            }

            // Get mean and covariance of selected component
            let mean = &means_vec[selected_component];
            let cov = &covs_vec[selected_component];

            // Generate sample from multivariate normal
            match self.covariance_type.as_str() {
                "spherical" => {
                    // For spherical, we just need to generate standard normal and scale
                    let sigma = cov[0][0].sqrt();
                    let mut sample = Vec::with_capacity(n_features);

                    for j in 0..n_features {
                        // Generate normal random variable: mean + sigma * standard_normal
                        let std_normal: f64 = random::<f64>() * 2.0 - 1.0; // Approximate normal
                        sample.push(mean[j] + sigma * std_normal);
                    }

                    samples.push(sample);
                },
                "diagonal" => {
                    // For diagonal, each dimension has its own variance
                    let mut sample = Vec::with_capacity(n_features);

                    for j in 0..n_features {
                        let sigma = cov[j][j].sqrt();
                        let std_normal: f64 = random::<f64>() * 2.0 - 1.0; // Approximate normal
                        sample.push(mean[j] + sigma * std_normal);
                    }

                    samples.push(sample);
                },
                "full" => {
                    // For full covariance, we need to generate multivariate normal
                    // 1. Perform Cholesky decomposition: cov = L * L^T
                    let chol = match self.rust_cholesky(cov) {
                        Ok(l) => l,
                        Err(_) => {
                            // Regularize and try again
                            let mut regularized = cov.clone();
                            for i in 0..n_features {
                                regularized[i][i] += 1e-6;
                            }
                            self.rust_cholesky(&regularized).unwrap()
                        }
                    };

                    // 2. Generate standard normal samples
                    let z: Vec<f64> = (0..n_features)
                        .map(|_| random::<f64>() * 2.0 - 1.0) // Approximate normal
                        .collect();

                    // 3. Transform: x = mean + L * z
                    let mut sample = mean.clone();
                    for i in 0..n_features {
                        for j in 0..=i { // Only need lower triangular part of L
                            sample[i] += chol[[i, j]] * z[j];
                        }
                    }

                    samples.push(sample);
                },
                _ => panic!("Unsupported covariance type"),
            }
        }

        samples
    }

    fn initialize_parameters(&mut self, data: &Array2<f64>) {
        let n_samples = data.shape()[0];
        let n_features = data.shape()[1];

        let weights = Array1::<f64>::ones(self.n_components) / self.n_components as f64;

        let mut means = Array2::<f64>::zeros((self.n_components, n_features));

        if self.init_method == "kmeans" {
            // TODO: Use kmeans over random points
            let indices: Vec<usize> = (0..n_samples)
                .choose_multiple(&mut thread_rng(), self.n_components)
                .into_iter()
                .collect();

            for (i, &idx) in indices.iter().enumerate() {
                let row = data.row(idx);
                for j in 0..n_features {
                    means[[i, j]] = row[j];
                }
            }
        } else if self.init_method == "random" {
            let mut rng = thread_rng();

            let mut min_vals = vec![f64::MAX; n_features];
            let mut max_vals = vec![f64::MIN; n_features];

            for i in 0..n_samples {
                for j in 0..n_features {
                    let val = data[[i, j]];
                    min_vals[j] = min_vals[j].min(val);
                    max_vals[j] = max_vals[j].max(val);
                }
            }

            for i in 0..self.n_components {
                for j in 0..n_features {
                    means[[i, j]] = rng.gen_range(min_vals[j]..=max_vals[j]);
                }
            }
        }

        let covariances = match self.covariance_type.as_str() {
            "full" => {
                let mut cov = Array2::<f64>::zeros((self.n_components * n_features, n_features));
                for i in 0..self.n_components {
                    for j in 0..n_features {
                        cov[[i * n_features + j, j]] = 1.0;
                    }
                }
                cov
            },
            "diagonal" => {
                Array2::<f64>::ones((self.n_components, n_features))
            },
            "spherical" => {
                Array2::<f64>::ones((self.n_components, 1))
            },
            _ => panic!("Unsupported covariance type: {}", self.covariance_type),
        };

        self.means = Some(means);
        self.covariances = Some(covariances);
        self.weights = Some(weights);
    }

    /// Expectation step: Calculate responsibilities
    fn e_step(&self, data: &Array2<f64>) -> (Array2<f64>, f64) {
        let n_samples = data.shape()[0];

        let weighted_log_prob = self.calculate_log_probabilities(&data);

        let mut log_prob_norm = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let row_slice = weighted_log_prob.slice(s![i, ..]);

            let max_value = row_slice.fold(f64::NEG_INFINITY, |a, &b| a.max(b));

            let sum_exp: f64 = row_slice
                .iter()
                .map(|&x| (x - max_value).exp())
                .sum();

            log_prob_norm.push(max_value + sum_exp.ln());
        }

        let responsibilities = self.calculate_responsibilities(&weighted_log_prob, &log_prob_norm);

        let log_likelihood = log_prob_norm.iter().sum();

        (responsibilities, log_likelihood)
    }

    /// Maximization step: Update parameters
    fn m_step(&mut self, data: &Array2<f64>, responsibilities: &Array2<f64>) {
        let n_samples = data.shape()[0];
        let n_features = data.shape()[1];

        // Update weights (mixing coefficients)
        let n_k: Array1<f64> = responsibilities.sum_axis(Axis(0));
        let new_weights = n_k.clone() / n_samples as f64;

        // Update means
        let mut new_means = Array2::<f64>::zeros((self.n_components, n_features));
        for k in 0..self.n_components {
            if n_k[k] > 0.0 {
                for i in 0..n_samples {
                    let resp_ik = responsibilities[[i, k]];
                    for j in 0..n_features {
                        new_means[[k, j]] += resp_ik * data[[i, j]];
                    }
                }
                for j in 0..n_features {
                    new_means[[k, j]] /= n_k[k];
                }
            }
        }

        let new_covariances = match self.covariance_type.as_str() {
            "full" => {
                let mut cov = Array2::<f64>::zeros((self.n_components * n_features, n_features));
                for k in 0..self.n_components {
                    if n_k[k] > 0.0 {
                        let mean_k = new_means.row(k).to_owned();
                        for i in 0..n_samples {
                            let resp_ik = responsibilities[[i, k]];
                            let x_i = data.row(i).to_owned();
                            let diff = &x_i - &mean_k;

                            for m in 0..n_features {
                                for n in 0..n_features {
                                    cov[[k * n_features + m, n]] += resp_ik * diff[m] * diff[n];
                                }
                            }
                        }
                        for m in 0..n_features {
                            for n in 0..n_features {
                                cov[[k * n_features + m, n]] /= n_k[k];
                            }
                        }

                        for m in 0..n_features {
                            cov[[k * n_features + m, m]] += 1e-6;
                        }
                    } else {
                        for m in 0..n_features {
                            cov[[k * n_features + m, m]] = 1.0;
                        }
                    }
                }
                cov
            },
            "diagonal" => {
                let mut cov = Array2::<f64>::zeros((self.n_components, n_features));
                for k in 0..self.n_components {
                    if n_k[k] > 0.0 {
                        let mean_k = new_means.row(k).to_owned();
                        for i in 0..n_samples {
                            let resp_ik = responsibilities[[i, k]];
                            let x_i = data.row(i).to_owned();
                            for j in 0..n_features {
                                let diff = x_i[j] - mean_k[j];
                                cov[[k, j]] += resp_ik * diff * diff;
                            }
                        }
                        for j in 0..n_features {
                            cov[[k, j]] /= n_k[k];
                            // Add regularization
                            cov[[k, j]] += 1e-6;
                        }
                    } else {
                        // If component is empty, use ones
                        for j in 0..n_features {
                            cov[[k, j]] = 1.0;
                        }
                    }
                }
                cov
            },
            "spherical" => {
                let mut cov = Array2::<f64>::zeros((self.n_components, 1));
                for k in 0..self.n_components {
                    if n_k[k] > 0.0 {
                        let mean_k = new_means.row(k).to_owned();
                        for i in 0..n_samples {
                            let resp_ik = responsibilities[[i, k]];
                            let x_i = data.row(i).to_owned();
                            let mut sq_diff_sum = 0.0;
                            for j in 0..n_features {
                                let diff = x_i[j] - mean_k[j];
                                sq_diff_sum += diff * diff;
                            }
                            cov[[k, 0]] += resp_ik * sq_diff_sum;
                        }
                        cov[[k, 0]] /= n_k[k] * n_features as f64;
                        cov[[k, 0]] += 1e-6;
                    } else {
                        cov[[k, 0]] = 1.0;
                    }
                }
                cov
            },
            _ => panic!("Unsupported covariance type"),
        };

        self.weights = Some(new_weights);
        self.means = Some(new_means);
        self.covariances = Some(new_covariances);
    }

    /// Calculate the multivariate normal density for a given data point
    fn multivariate_normal_pdf(&self, x: &Array1<f64>, mean: &Array1<f64>, cov: &Array2<f64>) -> f64 {
        let n = x.len() as f64;
        let diff = x - mean;

        let (det, quad_form) = match self.covariance_type.as_str() {
            "spherical" => {
                // In spherical case, cov is a scalar times identity
                let sigma = cov[[0, 0]];
                let det = sigma.powf(n);

                // For spherical, just calculate the dot product manually
                let quad_form = diff.iter().map(|&x| x * x).sum::<f64>() / sigma;
                (det, quad_form)
            },
            "diagonal" => {
                // det is product of diagonal elements
                let det = (0..cov.shape()[0]).map(|i| cov[[i, i]]).product::<f64>();

                // For diagonal, compute (x-μ)' Σ^-1 (x-μ) manually
                let mut quad_form = 0.0;
                for i in 0..diff.len() {
                    quad_form += diff[i] * diff[i] / cov[[i, i]];
                }
                (det, quad_form)
            },
            "full" => {
                // Implement Cholesky decomposition in pure Rust
                let chol_result = self.rust_cholesky(&array2_to_vec(cov));

                match chol_result {
                    Ok(chol) => {
                        // Determinant is square of product of diagonal elements
                        let diag_prod: f64 = (0..chol.shape()[0])
                            .map(|i| chol[[i, i]])
                            .product::<f64>();
                        let det = diag_prod * diag_prod;

                        // Solve for inv(L) * diff where L is the Cholesky factor
                        let y = self.rust_solve_triangular(&chol, &diff).unwrap();

                        // Quadratic form is ||y||^2 (dot product of y with itself)
                        let quad_form = y.iter().map(|&val| val * val).sum::<f64>();

                        (det, quad_form)
                    },
                    Err(_) => {
                        // Fallback to a regularized matrix
                        let eye = Array2::<f64>::eye(cov.shape()[0]);
                        let regularized = cov + &(&eye * 1e-6);
                        let chol = self.rust_cholesky(&array2_to_vec(&regularized)).unwrap();

                        // Determinant calculation
                        let diag_prod: f64 = (0..chol.shape()[0])
                            .map(|i| chol[[i, i]])
                            .product::<f64>();
                        let det = diag_prod * diag_prod;

                        // Solve and compute quadratic form
                        let y = self.rust_solve_triangular(&chol, &diff).unwrap();
                        let quad_form = y.iter().map(|&val| val * val).sum::<f64>();

                        (det, quad_form)
                    }
                }
            },
            _ => panic!("Unsupported covariance type"),
        };

        // PDF formula: (2π)^(-n/2) * |Σ|^(-1/2) * exp(-0.5 * quad_form)
        let normalizer = (2.0 * std::f64::consts::PI).powf(-n / 2.0) * det.powf(-0.5);
        normalizer * (-0.5 * quad_form).exp()
    }

    // Pure Rust implementation of Cholesky decomposition
    fn rust_cholesky(&self, a: &Vec<Vec<f64>>) -> Result<Array2<f64>, &'static str> {
        let n = a.len();
        if n == 0 || a[0].len() != n {
            return Err("Matrix must be square");
        }

        let mut l = Array2::<f64>::zeros((n, n));

        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0;

                if j == i {
                    // Diagonal elements
                    for k in 0..j {
                        sum += l[[j, k]] * l[[j, k]];
                    }
                    let val = a[j][j] - sum;
                    if val <= 0.0 {
                        return Err("Matrix is not positive definite");
                    }
                    l[[j, j]] = val.sqrt();
                } else {
                    // Off-diagonal elements
                    for k in 0..j {
                        sum += l[[i, k]] * l[[j, k]];
                    }
                    if l[[j, j]] == 0.0 {
                        return Err("Division by zero");
                    }
                    l[[i, j]] = (a[i][j] - sum) / l[[j, j]];
                }
            }
        }

        Ok(l)
    }

    // Pure Rust implementation of triangular solver (assumes lower triangular)
    fn rust_solve_triangular(&self, a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>, &'static str> {
        let n = a.shape()[0];
        if a.shape()[1] != n || b.len() != n {
            return Err("Dimension mismatch");
        }

        let mut x = Array1::<f64>::zeros(n);

        // Forward substitution for lower triangular
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..i {
                sum += a[[i, j]] * x[j];
            }

            if a[[i, i]] == 0.0 {
                return Err("Singular matrix");
            }

            x[i] = (b[i] - sum) / a[[i, i]];
        }

        Ok(x)
    }
    fn get_covariance_for_component(&self, k: usize, mean_k: &Array1<f64>, covariances: &Array2<f64>) -> Array2<f64> {
        match self.covariance_type.as_str() {
            "full" => {
                let start = k * mean_k.len();
                let end = (k + 1) * mean_k.len();
                covariances.slice(s![start..end, ..]).to_owned()
            },
            "diagonal" => {
                let diag_values = covariances.row(k).to_owned();
                Array2::from_diag(&diag_values)
            },
            "spherical" => {
                let sigma = covariances[[k, 0]];
                Array2::eye(mean_k.len()) * sigma
            },
            _ => panic!("Unsupported covariance type"),
        }
    }
    fn calculate_log_probabilities(&self, data: &Array2<f64>) -> Array2<f64> {
        let n_samples = data.nrows();
        let means = self.means.as_ref().unwrap();
        let covariances = self.covariances.as_ref().unwrap();
        let weights = self.weights.as_ref().unwrap();

        // Pre-compute log weights
        let log_weights: Vec<f64> = weights.iter().map(|&w| w.ln()).collect();

        let mut log_prob_matrix = Array2::<f64>::zeros((n_samples, self.n_components));

        // Pre-compute component covariances
        let component_covs: Vec<Array2<f64>> = (0..self.n_components)
            .map(|k| {
                let mean_k = means.row(k).to_owned();
                self.get_covariance_for_component(k, &mean_k, covariances)
            })
            .collect();

        for i in 0..n_samples {
            let x_i = data.row(i).to_owned();

            for k in 0..self.n_components {
                let mean_k = means.row(k).to_owned();
                let log_prob = self.multivariate_normal_pdf(&x_i, &mean_k, &component_covs[k]).ln();
                log_prob_matrix[[i, k]] = log_prob + log_weights[k];
            }
        }

        log_prob_matrix
    }

     fn calculate_responsibilities(&self, weighted_log_prob: &Array2<f64>, log_prob_norm: &[f64]) -> Array2<f64> {
        let n_samples = weighted_log_prob.nrows();

        // Create a column vector from log_prob_norm
        let log_norm_col = Array2::from_shape_fn((n_samples, 1), |(i, _)| log_prob_norm[i]);

        // Vectorized calculation of responsibilities
        (*&weighted_log_prob - &log_norm_col).mapv(f64::exp)
    }

    fn convert_to_array2f64(&self, data: &[Vec<D>]) -> Array2<f64> {
        let n_samples = data.len();
        let n_features = if n_samples > 0 { data[0].len() } else { 0 };

        let mut data_array = Array2::<f64>::zeros((n_samples, n_features));

        for (i, sample) in data.iter().enumerate() {
            for (j, &value) in sample.iter().enumerate() {
                data_array[[i, j]] = value.to_f64().expect("Cannot convert to f64");
            }
        }
        data_array
    }
}


fn log_sum_exp(row_slice: ArrayView1<f64>, log_prob_norm: &mut Vec<f64>) {
    let max_value = row_slice.fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    // Compute log-sum-exp
    let sum_exp: f64 = row_slice
        .iter()
        .map(|&x| (x - max_value).exp())
        .sum();

    // Append to log_prob_norm without returning it
    log_prob_norm.push(max_value + sum_exp.ln());
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;


    #[test]
    fn test_set_n_components() {
        let mut gm: GaussianMixture<f64, usize> = GaussianMixture::new();
        gm.set_n_components(3);
        assert_eq!(gm.n_components, 3);
    }

    #[test]
    fn test_set_max_iter() {
        let mut gm: GaussianMixture<f64, usize> = GaussianMixture::new();
        gm.set_max_iter(50);
        assert_eq!(gm.max_iter, 50);
    }

    #[test]
    fn test_set_tol() {
        let mut gm: GaussianMixture<f64, usize> = GaussianMixture::new();
        gm.set_tol(0.001);
        assert_relative_eq!(gm.tol, 0.001);
    }

    #[test]
    fn test_set_init_method() {
        let mut gm: GaussianMixture<f64, usize> = GaussianMixture::new();
        gm.set_init_method("random");
        assert_eq!(gm.init_method, "random");
    }


    #[test]
    fn test_fit_simple_clusters() {
        // Create two well-separated clusters
        let data = vec![
            // Cluster 1
            vec![0.1, 0.1], vec![0.2, 0.2], vec![0.0, 0.0], vec![0.1, 0.0], vec![0.2, 0.1],
            // Cluster 2
            vec![3.0, 3.0], vec![3.1, 3.1], vec![2.9, 2.9], vec![3.0, 2.9], vec![2.9, 3.0],
        ];
        let labels = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];

        let mut gm = GaussianMixture::new();
        gm.set_n_components(2);
        gm.set_init_method("kmeans");  // or any other initialization method
        gm.fit(data, labels);

        // After fitting, the means should be close to [0.1, 0.1] and [3.0, 3.0]
        let means = gm.means.unwrap();

        // Find which component corresponds to which cluster
        // (since the order might be arbitrary)
        let mut cluster1_idx = 0;
        let mut cluster2_idx = 1;

        if (means[[0, 0]] - 3.0).abs() < (means[[1, 0]] - 3.0).abs() {
            cluster1_idx = 1;
            cluster2_idx = 0;
        }

        assert_relative_eq!(means[[cluster1_idx, 0]], 0.15, epsilon = 0.3);
        assert_relative_eq!(means[[cluster1_idx, 1]], 0.1, epsilon = 0.3);
        assert_relative_eq!(means[[cluster2_idx, 0]], 3.0, epsilon = 0.3);
        assert_relative_eq!(means[[cluster2_idx, 1]], 3.0, epsilon = 0.3);

        // The weights should be approximately 0.5 each
        let weights = gm.weights.unwrap();
        assert_relative_eq!(weights[0], 0.5, epsilon = 0.1);
        assert_relative_eq!(weights[1], 0.5, epsilon = 0.1);
    }
    
    #[test]
    fn test_predict_proba() {
        // Create two well-separated clusters
        let train_data = vec![
            vec![0.0, 0.0], vec![0.1, 0.1],
            vec![3.0, 3.0], vec![3.1, 3.1],
        ];
        let train_labels = vec![0, 0, 1, 1];

        let mut gm = GaussianMixture::new();
        gm.set_n_components(2);
        gm.fit(train_data.clone(), train_labels);

        // Test a point right in the middle - should have some probability for each cluster
        let test_data = vec![vec![1.5, 1.5]];

        let probabilities = gm.predict_probability(test_data);

        // Probabilities should sum to approximately 1
        assert_relative_eq!(probabilities[[0, 0]] + probabilities[[0, 1]], 1.0, epsilon = 1e-5);

        // Test points very close to the centers
        let test_centers = vec![vec![0.05, 0.05], vec![3.05, 3.05]];
        let center_probs = gm.predict_probability(test_centers);

        // Find which component corresponds to which cluster
        let means = gm.means.unwrap();
        let mut cluster1_idx = 0;
        let mut cluster2_idx = 1;

        if (means[[0, 0]] - 3.0).abs() < (means[[1, 0]] - 3.0).abs() {
            cluster1_idx = 1;
            cluster2_idx = 0;
        }

        // Point near first center should have high probability for first cluster
        assert!(center_probs[[0, cluster1_idx]] > 0.9);

        // Point near second center should have high probability for second cluster
        assert!(center_probs[[1, cluster2_idx]] > 0.9);
    }

    #[test]
    fn test_score() {
        // Create a simple dataset
        let data = vec![
            vec![0.0, 0.0], vec![0.1, 0.1], vec![0.2, 0.2],
            vec![3.0, 3.0], vec![3.1, 3.1], vec![3.2, 3.2],
        ];
        let labels = vec![0, 0, 0, 1, 1, 1];

        let mut gm = GaussianMixture::new();
        gm.set_n_components(2);
        gm.fit(data.clone(), labels);

        // Score the same data
        let score1 = gm.score(data.clone().as_ref());

        // Score should be higher (less negative) for well-fitted data
        // than for random data
        let random_data = vec![
            vec![10.0, 10.0], vec![11.0, 11.0], vec![12.0, 12.0],
        ];
        let score2 = gm.score(random_data.as_ref());

        // Score for fitted data should be higher than for random data
        assert!(score1 > score2);
    }

    #[test]
    fn test_predict() {
        let mut gmm: GaussianMixture<f64, usize> = GaussianMixture::new();
        gmm.set_n_components(2);

        // Generate simple test data with two clearly separated clusters
        let data = vec![
            // Cluster 1 (centered around [0.0, 0.0])
            vec![0.1, 0.2],
            vec![-0.1, 0.1],
            vec![0.2, -0.2],
            vec![-0.2, -0.1],
            vec![0.0, 0.0],

            // Cluster 2 (centered around [5.0, 5.0])
            vec![4.9, 5.1],
            vec![5.1, 4.9],
            vec![4.8, 4.8],
            vec![5.2, 5.0],
            vec![5.0, 5.2],
        ];

        // Labels for the training data (0 for first cluster, 1 for second)
        let labels = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];

        // Fit the model to the training data
        gmm.fit(data.clone(), labels);

        // Test points clearly in cluster 1
        let test_cluster1 = vec![
            vec![0.3, 0.3],
            vec![-0.3, -0.3],
        ];

        // Test points clearly in cluster 2
        let test_cluster2 = vec![
            vec![4.7, 5.3],
            vec![5.3, 4.7],
        ];

        // Combine test points
        let test_data = [&test_cluster1[..], &test_cluster2[..]].concat();

        // Predict cluster assignments
        let predictions = gmm.predict(test_data);

        // We expect the first two test points to be assigned to cluster 0
        // and the second two test points to be assigned to cluster 1
        assert_eq!(predictions.len(), 4);

    let first_pair_same = predictions[0] == predictions[1];
    let second_pair_same = predictions[2] == predictions[3];
    let pairs_different = predictions[0] != predictions[2];
    assert!(first_pair_same && second_pair_same && pairs_different,
          "Cluster assignments aren't consistent: {:?}", predictions);

    }

    #[test]
    fn test_sample() {
        // We'll use a fixed RNG seed for deterministic testing
        let mut gm = GaussianMixture::new();
        gm.set_n_components(2);

        // Create and fit a simple model with known parameters
        let data = vec![
            vec![0.0, 0.0], vec![0.1, 0.1], vec![0.2, 0.2],
            vec![3.0, 3.0], vec![3.1, 3.1], vec![3.2, 3.2],
        ];
        let labels = vec![0, 0, 0, 1, 1, 1];

        gm.fit(data, labels);

        // Sample from the model
        let samples = gm.sample(100);

        // Check basic properties
        assert_eq!(samples.len(), 100);
        assert_eq!(samples[0].len(), 2);

        // Calculate mean of samples to check they're reasonable
        let mut means = vec![0.0, 0.0];
        for sample in &samples {
            means[0] += sample[0];
            means[1] += sample[1];
        }
        means[0] /= samples.len() as f64;
        means[1] /= samples.len() as f64;

        // The mean should be somewhere between our two clusters
        assert!(means[0] > 0.0 && means[0] < 3.2);
        assert!(means[1] > 0.0 && means[1] < 3.2);
    }

    #[test]
    fn test_convergence() {
        // Create a dataset with three clusters
        let data = vec![
            // Cluster 1
            vec![0.0, 0.0], vec![0.1, 0.1], vec![0.2, 0.2],
            // Cluster 2
            vec![3.0, 3.0], vec![3.1, 3.1], vec![3.2, 3.2],
            // Cluster 3
            vec![6.0, 0.0], vec![6.1, 0.1], vec![6.2, 0.2],
        ];
        let labels = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];

        // Test with different tolerance values
        let tols = [1e-3, 1e-6];
        let mut prev_score = f64::NEG_INFINITY;

        for &tol in &tols {
            let mut gm = GaussianMixture::new();
            gm.set_n_components(3);
            gm.set_tol(tol);
            gm.set_max_iter(100);
            gm.fit(data.clone(), labels.clone());

            let score = gm.score(data.clone().as_ref());

            // With tighter tolerance, score should be better or equal
            assert!(score >= prev_score);
            prev_score = score;
        }
    }

    #[test]
    fn test_different_covariance_types() {
        // Create a simple dataset
        let data = vec![
            vec![0.0, 0.0], vec![0.1, 0.1], vec![0.2, 0.2],
            vec![3.0, 3.0], vec![3.1, 3.1], vec![3.2, 3.2],
        ];
        let labels = vec![0, 0, 0, 1, 1, 1];

        let cov_types = ["full", "diagonal", "spherical"];

        for &cov_type in &cov_types {
            let mut gm = GaussianMixture::new();
            gm.set_n_components(2);
            gm.set_covariance_type(cov_type);

            // Should fit without errors
            gm.fit(data.clone(), labels.clone());

            // Should produce sensible predictions
            let predictions = gm.predict(data.clone());

            // Make sure all classes are represented in the predictions
            assert!(predictions.iter().any(|&x| x == 0));
            assert!(predictions.iter().any(|&x| x == 1));

            // For spherical covariance, check that the shape is correct
            if cov_type == "spherical" {
                let cov = gm.covariances.as_ref().unwrap();
                assert_eq!(cov.shape(), &[2, 1]);
            }

            // For diagonal covariance, check that the shape is correct
            if cov_type == "diagonal" {
                let cov = gm.covariances.as_ref().unwrap();
                assert_eq!(cov.shape(), &[2, 2]);
            }

            // For full covariance, check that the shape is correct
            if cov_type == "full" {
                let cov = gm.covariances.as_ref().unwrap();
                assert_eq!(cov.shape(), &[4, 2]);  // 2 components * 2 features = 4 rows
            }
        }
    }

    #[test]
    fn test_multivariate_normal_pdf() {
        use ndarray::{array};
        use std::f64::consts::PI;
        use approx::assert_relative_eq;

        // Create GMM with basic initialization to ensure it works
        let mut gm: GaussianMixture<f64, usize> = GaussianMixture::new();
        gm.set_n_components(2);
        gm.set_max_iter(100);
        gm.set_tol(1e-6);
        gm.set_covariance_type("full");

        // Test for 1D case
        let x1 = array![0.0];
        let mean1 = array![0.0];
        let cov1 = array![[1.0]];
        let pdf1 = gm.multivariate_normal_pdf(&x1, &mean1, &cov1);
        let expected1 = 1.0 / (2.0 * PI).sqrt();
        assert_relative_eq!(pdf1, expected1, epsilon = 1e-10);

        // Test for 2D case - standard normal
        let x2 = array![0.0, 0.0];
        let mean2 = array![0.0, 0.0];
        let cov2 = array![[1.0, 0.0], [0.0, 1.0]];
        let pdf2 = gm.multivariate_normal_pdf(&x2, &mean2, &cov2);
        let expected2 = 1.0 / (2.0 * PI);
        assert_relative_eq!(pdf2, expected2, epsilon = 1e-10);

        // Test with non-zero mean
        let x3 = array![1.0, 1.0];
        let mean3 = array![1.0, 1.0];
        let cov3 = array![[1.0, 0.0], [0.0, 1.0]];
        let pdf3 = gm.multivariate_normal_pdf(&x3, &mean3, &cov3);
        let expected3 = 1.0 / (2.0 * PI);
        assert_relative_eq!(pdf3, expected3, epsilon = 1e-10);

        // Test with correlation
        let x4 = array![0.0, 0.0];
        let mean4 = array![0.0, 0.0];
        let cov4 = array![[1.0, 0.5], [0.5, 1.0]];
        let pdf4 = gm.multivariate_normal_pdf(&x4, &mean4, &cov4);

        // Precisely calculated PDF value for the correlated normal at origin
        // Formula: 1/(2π * √|Σ|) where |Σ| = determinant of the covariance matrix
        // For this case, |Σ| = 1.0*1.0 - 0.5*0.5 = 0.75
        let determinant = 0.75;
        let expected4 = 1.0 / (2.0 * PI * determinant.sqrt());
        assert_relative_eq!(pdf4, expected4, epsilon = 1e-10);

        // Additional test: high-dimensional case
        let x5 = array![0.0, 0.0, 0.0];
        let mean5 = array![0.0, 0.0, 0.0];
        let cov5 = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let pdf5 = gm.multivariate_normal_pdf(&x5, &mean5, &cov5);
        let expected5 = 1.0 / (2.0 * PI).powf(3.0 / 2.0);
        assert_relative_eq!(pdf5, expected5, epsilon = 1e-10);

        // Edge case: point far from mean (should have very small probability)
        let x6 = array![10.0, 10.0];
        let mean6 = array![0.0, 0.0];
        let cov6 = array![[1.0, 0.0], [0.0, 1.0]];
        let pdf6 = gm.multivariate_normal_pdf(&x6, &mean6, &cov6);
        let expected6 = (1.0 / (2.0 * PI)) * (-100.0 / 2.0).exp();
        assert_relative_eq!(pdf6, expected6, epsilon = 1e-10);
    }
}
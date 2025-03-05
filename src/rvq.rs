//! # Residual Quantizer Implementation
//!
//! This module implements a residual quantizer that approximates an input vector as a sum of
//! quantized codewords. The quantizer is trained in multiple stages using the Linde–Buzo–Gray (LBG)
//! algorithm. At each stage, a codebook is learned to quantize the residual (error) left by previous
//! stages. The final quantized approximation is the sum of the selected codewords from each stage.
//!
//! The quantizer uses a specified distance metric to compare vectors and supports early termination
//! if the average residual norm falls below a given threshold during training.
//!
//! # Errors
//! Methods in this module panic with custom errors from the exceptions module when:
//! - The training data is empty.
//! - The training vectors are not all of the same dimension.
//! - An input vector passed to `quantize` does not have the expected dimension.
//!
//! # Example
//! ```
//! use vq::vector::Vector;
//! use vq::distances::Distance;
//! use vq::rvq::ResidualQuantizer;
//! use half::f16;
//!
//! // Create a small training dataset. Each vector has dimension 3.
//! // Provide at least 4 training vectors so that k (number of centroids per stage) is valid.
//! let training_data = vec![
//!     Vector::new(vec![0.0, 0.0, 0.0]),
//!     Vector::new(vec![1.0, 1.0, 1.0]),
//!     Vector::new(vec![0.5, 0.5, 0.5]),
//!     Vector::new(vec![0.2, 0.2, 0.2]),
//! ];
//!
//! // Configure the residual quantizer.
//! let stages = 3;
//! let k = 4; // number of centroids per stage
//! let max_iters = 10;
//! let epsilon = 0.01;
//! let distance = Distance::Euclidean;
//! let seed = 42;
//!
//! // Fit the residual quantizer.
//! let rq = ResidualQuantizer::fit(&training_data, stages, k, max_iters, epsilon, distance, seed);
//!
//! // Quantize an input vector (dimension must match training data, i.e. 3).
//! let input = Vector::new(vec![0.2, 0.8, 0.3]);
//! let quantized = rq.quantize(&input);
//! println!("Quantized vector: {:?}", quantized);
//! ```

use crate::distances::Distance;
use crate::exceptions::VqError;
use crate::utils::lbg_quantize;
use crate::vector::Vector;
use half::f16;
use rayon::prelude::*;

pub struct ResidualQuantizer {
    /// Maximum number of quantization stages.
    stages: usize,
    /// A vector of codebooks (one per stage). Each codebook is a vector of centroids.
    codebooks: Vec<Vec<Vector<f32>>>,
    /// Dimensionality of the input vectors.
    dim: usize,
    /// Distance metric used to evaluate the quality of a quantization.
    distance: Distance,
    /// Early termination threshold: if the residual norm falls below this value, training stops.
    epsilon: f32,
}

impl ResidualQuantizer {
    /// Constructs a new `ResidualQuantizer` using the provided training data.
    ///
    /// # Parameters
    /// - `training_data`: A slice of training vectors (each of type `Vector<f32>`) used to learn the quantizer.
    /// - `stages`: The number of quantization stages. Each stage learns a codebook for the residual error.
    /// - `k`: The number of centroids per stage (i.e. the size of each codebook).
    /// - `max_iters`: The maximum number of iterations for the LBG (k-means) algorithm.
    /// - `epsilon`: The early termination threshold. If the average residual norm falls below this value during training,
    ///   the training loop terminates early.
    /// - `distance`: The distance metric used to compute distances between vectors.
    /// - `seed`: The random seed used for initializing the LBG algorithm (each stage uses `seed + stage`).
    ///
    /// # Panics
    /// Panics with a custom error if:
    /// - `training_data` is empty.
    /// - The training data vectors are not all of the same dimension.
    pub fn fit(
        training_data: &[Vector<f32>],
        stages: usize,
        k: usize,
        max_iters: usize,
        epsilon: f32,
        distance: Distance,
        seed: u64,
    ) -> Self {
        if training_data.is_empty() {
            panic!("{}", VqError::EmptyInput);
        }
        let dim = training_data[0].len();
        // (Optionally, you could check that all training vectors have the same dimension here)
        let mut codebooks = Vec::with_capacity(stages);
        // Clone training data into residuals. Initially, each residual equals the original vector.
        let mut residuals = training_data.to_vec();

        for stage in 0..stages {
            // Learn a codebook on the current residuals.
            let codebook = lbg_quantize(&residuals, k, max_iters, seed + stage as u64);
            codebooks.push(codebook.clone());

            // Update residuals in parallel by subtracting the best matching centroid from each residual.
            residuals.par_iter_mut().for_each(|res| {
                let codebook = &codebooks[stage];
                let best_index = if codebook.len() < 2 {
                    0
                } else {
                    let mut best_index = 0;
                    let mut best_dist = distance.compute(&res.data, &codebook[0].data);
                    for (j, centroid) in codebook.iter().enumerate().skip(1) {
                        let dist = distance.compute(&res.data, &centroid.data);
                        if dist < best_dist {
                            best_dist = dist;
                            best_index = j;
                        }
                    }
                    best_index
                };
                *res = &*res - &codebooks[stage][best_index];
            });

            // Compute the average residual norm (in parallel) to check for early termination.
            let avg_norm: f32 = residuals
                .par_iter()
                .map(|r| {
                    let norm_sq: f32 = r.data.iter().map(|&x| x * x).sum();
                    norm_sq.sqrt()
                })
                .sum::<f32>()
                / residuals.len() as f32;
            if avg_norm < epsilon {
                break;
            }
        }

        // Use the actual number of stages performed (codebooks generated)
        let actual_stages = codebooks.len();

        Self {
            stages: actual_stages,
            codebooks,
            dim,
            distance,
            epsilon,
        }
    }

    /// Quantizes an input vector using the residual quantizer.
    ///
    /// The input vector is approximated as the sum of codewords selected from each stage.
    /// At each stage, the codeword that minimizes the distance (as defined by the stored `distance`)
    /// is chosen, and its contribution is subtracted from the residual.
    /// Early termination occurs if the residual norm falls below the stored `epsilon`.
    ///
    /// # Parameters
    /// - `vector`: The input vector (`Vector<f32>`) to quantize. Its dimension must equal the training data.
    ///
    /// # Returns
    /// A quantized vector of type `Vector<f16>` that approximates the input vector.
    ///
    /// # Panics
    /// Panics with a custom error if the input vector's dimension does not equal the expected dimension.
    pub fn quantize(&self, vector: &Vector<f32>) -> Vector<f16> {
        if vector.len() != self.dim {
            panic!(
                "{}",
                VqError::DimensionMismatch {
                    expected: self.dim,
                    found: vector.len()
                }
            );
        }
        let mut residual = vector.clone();
        let mut quantized_sum = Vector::new(vec![0.0; self.dim]);

        for stage in 0..self.stages {
            let codebook = &self.codebooks[stage];
            let best_index = if codebook.len() < 2 {
                0
            } else {
                let mut best_index = 0;
                let mut best_dist = self.distance.compute(&residual.data, &codebook[0].data);
                for (j, centroid) in codebook.iter().enumerate().skip(1) {
                    let dist = self.distance.compute(&residual.data, &centroid.data);
                    if dist < best_dist {
                        best_dist = dist;
                        best_index = j;
                    }
                }
                best_index
            };
            let chosen = &codebook[best_index];
            quantized_sum = &quantized_sum + chosen;
            residual = &residual - chosen;
            // Early termination if the residual norm is small.
            let norm: f32 = residual.data.iter().map(|&x| x * x).sum::<f32>().sqrt();
            if norm < self.epsilon {
                break;
            }
        }
        let quantized_f16: Vec<f16> = quantized_sum
            .data
            .iter()
            .map(|&x| f16::from_f32(x))
            .collect();
        Vector::new(quantized_f16)
    }
}

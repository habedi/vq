//! # Product Quantizer Implementation
//!
//! This module implements a Product Quantizer that partitions input vectors into sub-vectors,
//! then quantizes each sub-vector independently using a separate codebook learned via the
//! Linde-Buzo-Gray (LBG) algorithm. The final quantized representation is obtained by selecting
//! the best matching centroid (codeword) for each subspace using a specified distance metric,
//! and then concatenating these codewords (converted to half-precision, `f16`).
//!
//! # Errors
//! The `fit` and `quantize` methods panic with custom errors from the exceptions module when:
//! - The training data is empty.
//! - The dimension of the training vectors is less than `m` or not divisible by `m`.
//! - The input vector to `quantize` does not have the expected dimension.
//!
//! # Example
//! ```
//! use vq::vector::Vector;
//! use vq::distances::Distance;
//! use vq::pq::ProductQuantizer;
//!
//! // Create a small training dataset. Each vector has dimension 4.
//! let training_data = vec![
//!     Vector::new(vec![0.0, 0.0, 0.0, 0.0]),
//!     Vector::new(vec![1.0, 1.0, 1.0, 1.0]),
//!     Vector::new(vec![0.5, 0.5, 0.5, 0.5]),
//! ];
//!
//! // Partition the 4-dimensional vectors into m=2 subspaces (each of dimension 2).
//! let m = 2;
//! // Use k=2 centroids per subspace.
//! let k = 2;
//! let max_iters = 10;
//! let seed = 42;
//! let distance = Distance::Euclidean;
//!
//! // Fit the product quantizer with the training data.
//! let pq = ProductQuantizer::fit(&training_data, m, k, max_iters, distance, seed);
//!
//! // Quantize an input vector of dimension 4 (i.e. m * sub_dim = 2 * 2).
//! let input = Vector::new(vec![0.2, 0.8, 0.3, 0.7]);
//! let quantized = pq.quantize(&input);
//! println!("Quantized vector: {:?}", quantized);
//! ```

use crate::distances::Distance;
use crate::exceptions::VqError;
use crate::utils::lbg_quantize;
use crate::vector::Vector;
use half::f16;
use rayon::prelude::*;

pub struct ProductQuantizer {
    /// A vector of codebooks (one per subspace). Each codebook is a vector of centroids.
    codebooks: Vec<Vec<Vector<f32>>>,
    /// The dimensionality of each subspace (i.e. `n / m`).
    sub_dim: usize,
    /// The number of subspaces into which the input vector is partitioned.
    m: usize,
    /// The distance metric used for comparing subvectors with codebook centroids.
    distance: Distance,
}

impl ProductQuantizer {
    /// Constructs a new `ProductQuantizer` from training data.
    ///
    /// # Parameters
    /// - `training_data`: A slice of training vectors (`Vector<f32>`) used to learn the codebooks.
    /// - `m`: The number of subspaces into which the input vectors are partitioned.
    /// - `k`: The number of centroids (codewords) per subspace.
    /// - `max_iters`: The maximum number of iterations for the LBG (k-means) quantization algorithm.
    /// - `distance`: The distance metric used for comparing subvectors with codebook centroids.
    /// - `seed`: A random seed for initializing LBG quantization. Each subspace uses `seed + i`.
    ///
    /// # Panics
    /// Panics with a custom error if:
    /// - The training data is empty.
    /// - The dimension of the training vectors is less than `m`.
    /// - The dimension of the training vectors is not divisible by `m`.
    pub fn fit(
        training_data: &[Vector<f32>],
        m: usize,
        k: usize,
        max_iters: usize,
        distance: Distance,
        seed: u64,
    ) -> Self {
        if training_data.is_empty() {
            panic!("{}", VqError::EmptyInput);
        }
        let n = training_data[0].len();
        if n < m {
            panic!(
                "{}",
                VqError::InvalidParameter("Data dimension must be at least m".to_string())
            );
        }
        if n % m != 0 {
            panic!(
                "{}",
                VqError::InvalidParameter("Data dimension must be divisible by m".to_string())
            );
        }
        let sub_dim = n / m;

        // Learn a codebook for each subspace in parallel.
        let codebooks: Vec<Vec<Vector<f32>>> = (0..m)
            .into_par_iter()
            .map(|i| {
                // Extract the sub-training data for subspace `i`.
                let sub_training: Vec<Vector<f32>> = training_data
                    .iter()
                    .map(|v| {
                        let start = i * sub_dim;
                        let end = start + sub_dim;
                        Vector::new(v.data[start..end].to_vec())
                    })
                    .collect();
                // Learn a codebook for the subspace using LBG quantization.
                lbg_quantize(&sub_training, k, max_iters, seed + i as u64)
            })
            .collect();

        Self {
            codebooks,
            sub_dim,
            m,
            distance,
        }
    }

    /// Quantizes an input vector using the learned codebooks.
    ///
    /// The input vector is partitioned into `m` sub-vectors (each of dimension `sub_dim`).
    /// For each subspace, the best matching codeword is selected using the stored distance metric.
    /// The selected codewords are converted to half-precision (`f16`) and concatenated to form
    /// the final quantized representation.
    ///
    /// # Parameters
    /// - `vector`: The input vector (`Vector<f32>`) to be quantized.
    ///
    /// # Returns
    /// A quantized vector (`Vector<f16>`) representing the input vector.
    ///
    /// # Panics
    /// Panics with a custom error if the input vector's dimension does not equal `m * sub_dim`.
    pub fn quantize(&self, vector: &Vector<f32>) -> Vector<f16> {
        let n = vector.len();
        if n != self.sub_dim * self.m {
            panic!(
                "{}",
                VqError::DimensionMismatch {
                    expected: self.sub_dim * self.m,
                    found: n
                }
            );
        }

        // Process each subspace in parallel to quantize the corresponding sub-vector.
        let quantized_subs: Vec<Vec<f16>> = (0..self.m)
            .into_par_iter()
            .map(|i| {
                let start = i * self.sub_dim;
                let end = start + self.sub_dim;
                let sub_vector = &vector.data[start..end];
                let codebook = &self.codebooks[i];
                let mut best_index = 0;
                let mut best_dist = self.distance.compute(sub_vector, &codebook[0].data);
                for (j, centroid) in codebook.iter().enumerate().skip(1) {
                    let dist = self.distance.compute(sub_vector, &centroid.data);
                    if dist < best_dist {
                        best_dist = dist;
                        best_index = j;
                    }
                }
                // Convert the chosen centroid's sub-vector from f32 to f16.
                codebook[best_index]
                    .data
                    .iter()
                    .map(|&val| f16::from_f32(val))
                    .collect()
            })
            .collect();

        // Flatten the quantized sub-vectors into one contiguous vector.
        let quantized_data: Vec<f16> = quantized_subs.into_iter().flatten().collect();
        Vector::new(quantized_data)
    }
}

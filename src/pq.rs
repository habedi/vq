use crate::distances::Distance;
use crate::utils::lbg_quantize;
use crate::vector::Vector;
use half::f16;
use rayon::prelude::*;

/// A Product Quantizer that partitions input vectors into sub-vectors and quantizes each
/// sub-vector independently using a separate codebook.
///
/// The quantizer splits an input vector of dimension `n` into `m` sub-vectors (each of dimension
/// `sub_dim = n / m`). For each subspace, a codebook is learned from training data using the LBG
/// (k-means) algorithm. During quantization, the best matching centroid is chosen from each
/// subspace using the stored distance metric, and the final quantized representation is the
/// concatenation of these centroids, converted to half-precision (`f16`).
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
    /// - `seed`: A random seed for initializing LBG quantization. Each subspace uses `seed + i`.
    /// - `distance`: The distance metric used for comparing subvectors with codebook centroids.
    ///
    /// # Panics
    /// Panics if:
    /// - The training data is empty.
    /// - The dimension of the training vectors is less than `m` or not divisible by `m`.
    pub fn new(
        training_data: &[Vector<f32>],
        m: usize,
        k: usize,
        max_iters: usize,
        distance: Distance,
        seed: u64,
    ) -> Self {
        assert!(!training_data.is_empty(), "Training data must not be empty");
        let n = training_data[0].len();
        assert!(n >= m, "Data dimension must be at least m");
        assert_eq!(n % m, 0, "Data dimension must be divisible by m");
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
    /// Panics if the dimension of the input vector does not equal `m * sub_dim`.
    pub fn quantize(&self, vector: &Vector<f32>) -> Vector<f16> {
        let n = vector.len();
        assert_eq!(
            n,
            self.sub_dim * self.m,
            "Input vector has incorrect dimension"
        );

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

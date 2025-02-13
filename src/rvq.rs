use crate::distances::Distance;
use crate::utils::lbg_quantize;
use crate::vector::Vector;
use half::f16;
use rayon::prelude::*;

/// A residual quantizer that approximates an input vector as a sum of quantized codewords.
///
/// The quantizer is trained in multiple stages using the Linde–Buzo–Gray (LBG) algorithm.
/// In each stage, it learns a codebook to quantize the residual (error) left by the previous stages.
/// The final quantized approximation is the sum of the codewords selected from each stage.
///
/// The distance metric used for comparing vectors and the early termination threshold (epsilon)
/// are specified at construction time and used during both training and quantization.
pub struct ResidualQuantizer {
    /// Maximum number of quantization stages.
    stages: usize,
    /// A vector of codebooks (one per stage). Each codebook is a vector of centroids.
    codebooks: Vec<Vec<Vector<f32>>>,
    /// Dimensionality of the input vectors.
    dim: usize,
    /// Distance metric used to evaluate the quality of a quantization.
    distance: Distance,
    /// Early termination threshold: if the residual norm falls below this value, training (or quantization) stops.
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
    /// This function panics if:
    /// - `training_data` is empty.
    /// - The training data vectors are not all of the same dimension.
    pub fn fit(
        training_data: &[Vector<f32>],
        stages: usize,
        k: usize,
        max_iters: usize,
        epsilon: f32, // early termination threshold
        distance: Distance,
        seed: u64,
    ) -> Self {
        assert!(!training_data.is_empty(), "Training data cannot be empty");
        let dim = training_data[0].len();
        let mut codebooks = Vec::with_capacity(stages);
        // Clone training data into residuals. Initially, each residual equals the original vector.
        let mut residuals = training_data.to_vec();

        for stage in 0..stages {
            // Learn a codebook on the current residuals.
            let codebook = lbg_quantize(&residuals, k, max_iters, seed + stage as u64);
            codebooks.push(codebook.clone());

            // Update residuals in parallel by subtracting the best matching centroid from each residual.
            residuals.par_iter_mut().for_each(|res| {
                let mut best_index = 0;
                let mut best_dist = distance.compute(&res.data, &codebooks[stage][0].data);
                for (j, centroid) in codebooks[stage].iter().enumerate().skip(1) {
                    let dist = distance.compute(&res.data, &centroid.data);
                    if dist < best_dist {
                        best_dist = dist;
                        best_index = j;
                    }
                }
                // Subtract the chosen centroid from the residual.
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

        Self {
            stages,
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
    /// - `vector`: The input vector (`Vector<f32>`) to quantize. Its dimension must match the training data.
    ///
    /// # Returns
    /// A quantized vector of type `Vector<f16>` that approximates the input vector.
    ///
    /// # Panics
    /// Panics if the input vector's dimension does not equal the expected dimension.
    pub fn quantize(&self, vector: &Vector<f32>) -> Vector<f16> {
        assert_eq!(vector.len(), self.dim, "Input vector has wrong dimension");
        let mut residual = vector.clone();
        let mut quantized_sum = Vector::new(vec![0.0; self.dim]);

        for stage in 0..self.stages {
            let codebook = &self.codebooks[stage];
            let mut best_index = 0;
            let mut best_dist = self.distance.compute(&residual.data, &codebook[0].data);
            for (j, centroid) in codebook.iter().enumerate().skip(1) {
                let dist = self.distance.compute(&residual.data, &centroid.data);
                if dist < best_dist {
                    best_dist = dist;
                    best_index = j;
                }
            }
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

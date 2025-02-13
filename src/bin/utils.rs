#![allow(dead_code)]

use anyhow::Result;
use rand::prelude::*;
use rand_distr::{Distribution, Uniform};
use rayon::prelude::*;
use std::collections::HashSet;
use vq::vector::Vector;

// Benchmark parameters.
pub const SEED: u64 = 66; // Seed for random number generation.
pub const NUM_SAMPLES: [usize; 5] = [1_000, 5_000, 10_000, 50_000, 100_000]; // Number of samples (vectors) to generate.
pub const DIM: usize = 128; // Dimensionality of the data (vector length).
pub const M: usize = 16; // Number of subspaces to partition the data into.
pub const K: usize = 256; // Number of centroids per subspace.
pub const MAX_ITERS: usize = 10; // Maximum number of LBG iterations.

/// Structure to hold benchmark metrics.
#[derive(serde::Serialize)]
pub struct BenchmarkResult {
    pub n_samples: usize,
    pub n_dims: usize,
    pub training_time_ms: f64,
    pub quantization_time_ms: f64,
    pub reconstruction_error: f32,
    pub recall: f32,
    pub memory_reduction_ratio: f32,
}

/// Generate synthetic data as a `Vec<Vector<f32>>` using the given seed.
pub fn generate_synthetic_data(n_samples: usize, n_dims: usize, seed: u64) -> Vec<Vector<f32>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    // Unwrap is safe here because the distribution creation is unlikely to fail.
    let uniform = Uniform::new(0.0, 1.0).unwrap();
    (0..n_samples)
        .map(|_| {
            let data: Vec<f32> = (0..n_dims).map(|_| uniform.sample(&mut rng)).collect();
            Vector::new(data)
        })
        .collect()
}

/// Compute the Euclidean distance between two vectors.
pub fn euclidean_distance(a: &Vector<f32>, b: &Vector<f32>) -> f32 {
    a.distance2(b).sqrt()
}

/// Compute the mean squared reconstruction error between original and reconstructed vectors.
/// This version uses parallel iterators for improved performance.
pub fn calculate_reconstruction_error(
    original: &[Vector<f32>],
    reconstructed: &[Vector<f32>],
) -> f32 {
    let total_elements = (original.len() * original[0].len()) as f32;
    let sum_error: f32 = original
        .par_iter()
        .zip(reconstructed.par_iter())
        .map(|(o, r)| {
            o.data
                .iter()
                .zip(r.data.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f32>()
        })
        .sum();
    sum_error / total_elements
}

/// Compute recall@k by comparing the nearest neighbors in the original and reconstructed spaces.
pub fn calculate_recall(original: &[Vector<f32>], approx: &[Vector<f32>], k: usize) -> Result<f32> {
    let n_samples = original.len();
    let max_eval_samples = 1000;
    let eval_samples = if n_samples > max_eval_samples {
        max_eval_samples
    } else {
        n_samples
    };
    let step = if n_samples / eval_samples > 0 {
        n_samples / eval_samples
    } else {
        1
    };
    let mut total_recall = 0.0;

    for i in (0..n_samples).step_by(step) {
        let query = &original[i];
        let search_window = if n_samples > 10_000 { 5000 } else { n_samples };

        let start_idx = if i > search_window / 2 {
            i - search_window / 2
        } else {
            0
        };
        let end_idx = (i + search_window / 2).min(n_samples);

        // True neighbors using original data.
        let mut true_neighbors: Vec<(usize, f32)> = (start_idx..end_idx)
            .filter(|&j| j != i)
            .map(|j| (j, euclidean_distance(query, &original[j])))
            .collect();
        true_neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let true_neighbors: Vec<usize> =
            true_neighbors.iter().take(k).map(|&(idx, _)| idx).collect();

        // Approximate neighbors using reconstructed data.
        let mut approx_neighbors: Vec<(usize, f32)> = (start_idx..end_idx)
            .filter(|&j| j != i)
            .map(|j| (j, euclidean_distance(&approx[i], &approx[j])))
            .collect();
        approx_neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let approx_neighbors: Vec<usize> = approx_neighbors
            .iter()
            .take(k)
            .map(|&(idx, _)| idx)
            .collect();

        // Use a HashSet for faster intersection lookup.
        let approx_set: HashSet<_> = approx_neighbors.into_iter().collect();
        let intersection = true_neighbors
            .iter()
            .filter(|&&idx| approx_set.contains(&idx))
            .count() as f32;
        total_recall += intersection / k as f32;
    }

    Ok(total_recall / (n_samples / step) as f32)
}

fn main() {}

//! # Utility Functions for Vq
//!
//! This module contains helper functions for vector quantization.
//! The main function here is `lbg_quantize`, which implements the Linde-Buzo-Gray (LBG)
//! algorithm for vector quantization using parallel operations when it is beneficial.

use crate::exceptions::VqError;
use crate::vector::{mean_vector, Vector};
use rand::prelude::IndexedRandom;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;

/// Quantizes the input data into `k` clusters using the LBG algorithm.
///
/// The function randomly selects `k` initial centroids and iteratively refines them by
/// assigning each data point to the nearest centroid and then recomputing the centroids.
/// Parallel iteration is used for assignments and cluster grouping when possible.
///
/// # Parameters
/// - `data`: A slice of vectors to quantize.
/// - `k`: The number of clusters (must be > 0 and ≤ number of data points).
/// - `max_iters`: Maximum iterations for the refinement process.
/// - `seed`: A seed for random number generation to ensure reproducibility.
///
/// # Returns
/// A vector of centroids (quantized vectors).
///
/// # Panics
/// - If `k` is 0.
/// - If there are fewer data points than clusters.
pub fn lbg_quantize(
    data: &[Vector<f32>],
    k: usize,
    max_iters: usize,
    seed: u64,
) -> Vec<Vector<f32>> {
    let n = data.len();
    if k == 0 {
        panic!(
            "{}",
            VqError::InvalidParameter("k must be greater than 0".to_string())
        );
    }
    if n < k {
        panic!(
            "{}",
            VqError::InvalidParameter("Not enough data points for k clusters".to_string())
        );
    }

    let mut rng = StdRng::seed_from_u64(seed);
    // Randomly select k initial centroids.
    let mut centroids: Vec<Vector<f32>> = data.choose_multiple(&mut rng, k).cloned().collect();
    let mut assignments = vec![0; n];

    for _ in 0..max_iters {
        // Assignment step: assign each vector to the nearest centroid.
        let new_assignments: Vec<usize> = data
            .par_iter()
            .map(|v| {
                let mut best = 0;
                let mut best_dist = v.distance2(&centroids[0]);
                for (j, centroid) in centroids.iter().enumerate().skip(1) {
                    let dist = v.distance2(centroid);
                    if dist < best_dist {
                        best = j;
                        best_dist = dist;
                    }
                }
                best
            })
            .collect();

        // Check if any assignment changed.
        let changed = new_assignments
            .iter()
            .zip(assignments.iter())
            .any(|(new, old)| new != old);
        assignments = new_assignments;

        // Update step: group data points into clusters.
        let clusters: Vec<Vec<Vector<f32>>> = (0..k)
            .into_par_iter()
            .map(|cluster_idx| {
                data.iter()
                    .zip(assignments.iter())
                    .filter(|(_, &assign)| assign == cluster_idx)
                    .map(|(v, _)| v.clone())
                    .collect::<Vec<_>>()
            })
            .collect();

        // Recompute centroids for each cluster.
        for j in 0..k {
            if !clusters[j].is_empty() {
                centroids[j] = mean_vector(&clusters[j]);
            } else {
                // Reinitialize an empty cluster with a random data point.
                centroids[j] = data.choose(&mut rng).unwrap().clone();
            }
        }

        if !changed {
            break;
        }
    }
    centroids
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::Vector;

    /// Create test data.
    fn get_data() -> Vec<Vector<f32>> {
        vec![
            Vector::new(vec![1.0, 2.0]),
            Vector::new(vec![2.0, 3.0]),
            Vector::new(vec![3.0, 4.0]),
            Vector::new(vec![4.0, 5.0]),
        ]
    }

    #[test]
    fn lbg_quantize_basic_functionality() {
        let data = get_data();
        let centroids = lbg_quantize(&data, 2, 10, 42);
        assert_eq!(centroids.len(), 2);
    }

    #[test]
    #[should_panic(expected = "k must be greater than 0")]
    fn lbg_quantize_k_zero() {
        let data = vec![Vector::new(vec![1.0, 2.0]), Vector::new(vec![2.0, 3.0])];
        lbg_quantize(&data, 0, 10, 42);
    }

    #[test]
    #[should_panic(expected = "Not enough data points for k clusters")]
    fn lbg_quantize_not_enough_data_points() {
        let data = vec![Vector::new(vec![1.0, 2.0])];
        lbg_quantize(&data, 2, 10, 42);
    }

    #[test]
    fn lbg_quantize_single_data_point() {
        let data = vec![Vector::new(vec![1.0, 2.0])];
        let centroids = lbg_quantize(&data, 1, 10, 42);
        assert_eq!(centroids.len(), 1);
        assert_eq!(centroids[0], Vector::new(vec![1.0, 2.0]));
    }

    #[test]
    fn lbg_quantize_multiple_iterations() {
        let data = get_data();
        let centroids = lbg_quantize(&data, 2, 100, 42);
        assert_eq!(centroids.len(), 2);
    }
}

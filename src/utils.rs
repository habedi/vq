use crate::vector::{mean_vector, Vector};
use rand::prelude::IndexedRandom;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;

pub fn lbg_quantize(
    data: &[Vector<f32>],
    k: usize,
    max_iters: usize,
    seed: u64,
) -> Vec<Vector<f32>> {
    let n = data.len();
    // Ensure k is greater than 0 and that there are at least k data points.
    assert!(k > 0, "k must be greater than 0");
    assert!(n >= k, "Not enough data points for k clusters");

    let mut rng = StdRng::seed_from_u64(seed);
    // Randomly select k initial centroids from the data.
    let mut centroids: Vec<Vector<f32>> = data.choose_multiple(&mut rng, k).cloned().collect();
    // Initialize cluster assignments.
    let mut assignments = vec![0; n];

    for _ in 0..max_iters {
        // Assignment step: for each data point, find the nearest centroid in parallel.
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

        // Check if any assignment has changed.
        let changed = new_assignments
            .iter()
            .zip(assignments.iter())
            .any(|(new, old)| new != old);
        assignments = new_assignments;

        // Update step: group data points into clusters in parallel.
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

    // Helper function to create the test data.
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

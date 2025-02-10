use crate::vector::{mean_vector, Vector};
use rand::prelude::IndexedRandom;
use rand::rngs::StdRng;
use rand::SeedableRng;

pub fn lbg_quantize(
    data: &[Vector<f32>],
    k: usize,
    max_iters: usize,
    seed: u64,
) -> Vec<Vector<f32>> {
    let n = data.len();
    assert!(n >= k, "Not enough data points for k clusters");
    let mut rng = StdRng::seed_from_u64(seed);
    let mut centroids: Vec<Vector<f32>> = data.choose_multiple(&mut rng, k).cloned().collect();
    let mut assignments = vec![0; n];

    for _ in 0..max_iters {
        let mut changed = false;
        for (i, v) in data.iter().enumerate() {
            let mut best = 0;
            let mut best_dist = v.distance2(&centroids[0]);
            for (j, centroid) in centroids.iter().enumerate().skip(1) {
                let dist = v.distance2(centroid);
                if dist < best_dist {
                    best_dist = dist;
                    best = j;
                }
            }
            if assignments[i] != best {
                assignments[i] = best;
                changed = true;
            }
        }
        let mut clusters: Vec<Vec<Vector<f32>>> = vec![Vec::new(); k];
        for (i, &cluster_idx) in assignments.iter().enumerate() {
            clusters[cluster_idx].push(data[i].clone());
        }
        for j in 0..k {
            if !clusters[j].is_empty() {
                centroids[j] = mean_vector(&clusters[j]);
            }
        }
        if !changed {
            break;
        }
    }
    centroids
}

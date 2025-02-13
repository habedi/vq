#![allow(dead_code)]

use vq::vector::Vector;

pub const NUM_VECTORS: usize = 100;
pub const BENCH_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);
pub const DIM: usize = 64;
pub const M: usize = 4;
pub const K: usize = 8;
pub const MAX_ITERS: usize = 10;
pub const SEED: u64 = 42;

/// Generates a synthetic training dataset of `num` vectors, each of dimension `dim`.
pub fn generate_training_data(num: usize, dim: usize) -> Vec<Vector<f32>> {
    (0..num)
        .map(|_| {
            let data: Vec<f32> = (0..dim).map(|i| (i as f32) / (dim as f32)).collect();
            Vector::new(data)
        })
        .collect()
}

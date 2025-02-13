#![allow(dead_code)]

use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use vq::vector::Vector;

pub const SEED: u64 = 42;
pub const MIN_VAL: f32 = -1000.0;
pub const MAX_VAL: f32 = 1000.0;

pub fn seeded_rng() -> StdRng {
    StdRng::seed_from_u64(SEED)
}

fn generate_random_vector<R: Rng>(rng: &mut R, dim: usize) -> Vector<f32> {
    let data: Vec<f32> = (0..dim)
        .map(|_| rng.random_range(MIN_VAL..MAX_VAL))
        .collect();
    Vector::new(data)
}

pub fn generate_test_data<R: Rng>(rng: &mut R, n: usize, dim: usize) -> Vec<Vector<f32>> {
    (0..n).map(|_| generate_random_vector(rng, dim)).collect()
}

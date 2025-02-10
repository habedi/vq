#[path = "utils.rs"]
mod utils;

use half::f16;
use utils::{generate_test_data, seeded_rng};
use vq::distances::Distance;
use vq::rvq::ResidualQuantizer;

#[test]
fn test_rvq_dimension() {
    let mut rng = seeded_rng();
    let training_data = generate_test_data(&mut rng, 1000, 10);
    let stages = 3;
    let k = 2;
    let max_iters = 50;
    let seed = 42;
    let rvq = ResidualQuantizer::new(
        &training_data,
        stages,
        k,
        max_iters,
        seed,
        Distance::SquaredEuclidean,
    );
    for vector in training_data.iter() {
        let quantized = rvq.quantize(vector, Distance::SquaredEuclidean);
        assert_eq!(quantized.len(), vector.len());
    }
}

#[test]
fn test_rvq_reconstruction_error() {
    let mut rng = seeded_rng();
    let training_data = generate_test_data(&mut rng, 1000, 10);
    let stages = 3;
    let k = 2;
    let max_iters = 50;
    let seed = 42;
    let rvq = ResidualQuantizer::new(
        &training_data,
        stages,
        k,
        max_iters,
        seed,
        Distance::SquaredEuclidean,
    );
    for vector in training_data.iter() {
        let quantized = rvq.quantize(vector, Distance::SquaredEuclidean);
        let reconstructed: Vec<f32> = quantized.data.iter().map(|&x| f16::to_f32(x)).collect();
        let total_error: f32 = vector
            .data
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(total_error.is_finite());
    }
}

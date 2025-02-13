#[path = "utils.rs"]
mod utils;

use half::f16;
use utils::{generate_test_data, seeded_rng};
use vq::distances::Distance;
use vq::opq::OptimizedProductQuantizer;

#[test]
fn test_opq_dimension() {
    let mut rng = seeded_rng();
    // Generate 1000 training vectors of dimension 10.
    let training_data = generate_test_data(&mut rng, 1000, 10);
    let m = 2; // Must divide dimension (10) evenly.
    let k = 2;
    let max_iters = 50;
    let opq_iters = 5;
    let seed = 42;
    let opq = OptimizedProductQuantizer::new(
        &training_data,
        m,
        k,
        max_iters,
        opq_iters,
        Distance::SquaredEuclidean,
        seed,
    );
    // For each training vector, quantization should yield a vector of the same dimension.
    for vector in training_data.iter() {
        let quantized = opq.quantize(vector);
        assert_eq!(
            quantized.len(),
            vector.len(),
            "Quantized vector length should match input dimension"
        );
    }
}

#[test]
fn test_opq_reconstruction_error() {
    let mut rng = seeded_rng();
    // Generate 1000 training vectors of dimension 10.
    let training_data = generate_test_data(&mut rng, 1000, 10);
    let m = 2;
    let k = 2;
    let max_iters = 50;
    let opq_iters = 5;
    let seed = 42;
    let opq = OptimizedProductQuantizer::new(
        &training_data,
        m,
        k,
        max_iters,
        opq_iters,
        Distance::SquaredEuclidean,
        seed,
    );
    // For each training vector, quantize and compute total absolute error.
    for vector in training_data.iter() {
        let quantized = opq.quantize(vector);
        let reconstructed: Vec<f32> = quantized.data.iter().map(|&x| f16::to_f32(x)).collect();
        let total_error: f32 = vector
            .data
            .iter()
            .zip(reconstructed.iter())
            .map(|(orig, recon)| (orig - recon).abs())
            .sum();
        assert!(
            total_error.is_finite(),
            "Total reconstruction error {} is not finite",
            total_error
        );
        // Optionally, check that the error is below some acceptable threshold.
        // assert!(
        //     total_error < 1e-2,
        //     "Total reconstruction error {} too high",
        //     total_error
        // );
    }
}

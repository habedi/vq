#[path = "utils.rs"]
mod utils;

use half::f16;
use utils::{generate_test_data, seeded_rng};
use vq::distances::Distance;
use vq::pq::ProductQuantizer;

#[test]
fn test_pq_on_random_vectors() {
    let mut rng = seeded_rng();
    let training_data = generate_test_data(&mut rng, 1000, 10);
    let m = 2;
    let k = 2;
    let max_iters = 50;
    let seed = 42;
    let pq = ProductQuantizer::new(
        &training_data,
        m,
        k,
        max_iters,
        Distance::SquaredEuclidean,
        seed,
    );
    for vector in training_data.iter() {
        let quantized = pq.quantize(vector);
        assert_eq!(
            quantized.len(),
            vector.len(),
            "Quantized vector length should match input dimension"
        );
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
    }
}

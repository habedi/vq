#[path = "utils.rs"]
mod utils;

use half::f16;
use utils::{generate_test_data, seeded_rng};
use vq::distances::Distance;
use vq::tsvq::TSVQ;
use vq::vector::Vector;

#[test]
fn test_tsvq_on_identical_vectors() {
    let training_vector = Vector::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let training_data = vec![training_vector.clone(); 10];
    let max_depth = 3;
    let tsvq = TSVQ::new(&training_data, max_depth);
    let quantized = tsvq.quantize(&training_vector, Distance::SquaredEuclidean);
    assert_eq!(quantized.len(), training_vector.len());
    let reconstructed: Vec<f32> = quantized.data.iter().map(|&x| f16::to_f32(x)).collect();
    for (orig, recon) in training_vector.data.iter().zip(reconstructed.iter()) {
        assert!((orig - recon).abs() < 1e-6);
    }
}

#[test]
fn test_tsvq_on_random_vectors() {
    let mut rng = seeded_rng();
    let training_data = generate_test_data(&mut rng, 1000, 10);
    let max_depth = 3;
    let tsvq = TSVQ::new(&training_data, max_depth);
    for vector in training_data.iter() {
        let quantized = tsvq.quantize(vector, Distance::SquaredEuclidean);
        assert_eq!(quantized.len(), vector.len());
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

#[path = "utils.rs"]
mod utils;

use utils::{generate_test_data, seeded_rng};
use vq::sq::ScalarQuantizer;
use vq::vector::Vector;

#[test]
fn test_scalar_quantizer_on_scalars() {
    let quantizer = ScalarQuantizer::fit(-1.0, 1.0, 5);
    let test_values = vec![-1.2, -1.0, -0.8, -0.3, 0.0, 0.3, 0.6, 1.0, 1.2];
    for x in test_values {
        let vec_x = Vector::new(vec![x]);
        let indices = quantizer.quantize(&vec_x);
        assert_eq!(indices.len(), 1, "Expected one index per scalar vector");
        let reconstructed = quantizer.min + indices.data[0] as f32 * quantizer.step;
        let clamped = if x < -1.0 {
            -1.0
        } else if x > 1.0 {
            1.0
        } else {
            x
        };
        let error = (reconstructed - clamped).abs();
        let max_error = quantizer.step / 2.0;
        assert!(
            error <= max_error + 1e-6,
            "Reconstruction error {} too high for input {}",
            error,
            x
        );
    }
}

#[test]
fn test_scalar_quantizer_on_large_vectors() {
    let mut rng = seeded_rng();
    let n = 100;
    let dim = 1024;
    let data = generate_test_data(&mut rng, n, dim);
    let quantizer = ScalarQuantizer::fit(-1000.0, 1000.0, 256);

    for vector in data.iter() {
        let indices = quantizer.quantize(vector);
        assert_eq!(
            indices.len(),
            dim,
            "Quantized indices length should match input dimension"
        );
        let reconstructed: Vec<f32> = indices
            .data
            .iter()
            .map(|&i| quantizer.min + i as f32 * quantizer.step)
            .collect();
        assert_eq!(
            reconstructed.len(),
            dim,
            "Reconstructed vector length should match input dimension"
        );
        for (&orig, &recon) in vector.data.iter().zip(reconstructed.iter()) {
            let clamped = if orig < quantizer.min {
                quantizer.min
            } else if orig > quantizer.max {
                quantizer.max
            } else {
                orig
            };
            let error = (clamped - recon).abs();
            assert!(
                error <= quantizer.step / 2.0 + 1e-6,
                "Error {} exceeds allowed maximum for original {}",
                error,
                orig
            );
        }
    }
}

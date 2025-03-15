#[path = "utils.rs"]
mod utils;

use utils::{generate_test_data, seeded_rng};
use vq::bq::BinaryQuantizer;
use vq::vector::Vector;

#[test]
fn test_binary_quantizer_large_vector() {
    let mut rng = seeded_rng();
    let dim = 1024;
    let data = generate_test_data(&mut rng, 1, dim);
    let vector = &data[0];

    // Using threshold 0.0, low = 0, high = 1 (unsigned 8-bit values)
    let quantizer = BinaryQuantizer::fit(0.0, 0, 1);
    let quantized: Vector<u8> = quantizer.quantize(vector);
    assert_eq!(
        quantized.len(),
        dim,
        "Quantized output must have same dimension as input"
    );

    for (i, &val) in quantized.data.iter().enumerate() {
        let expected = if vector.data[i] >= 0.0 { 1 } else { 0 };
        assert_eq!(
            val, expected,
            "At index {}: expected {} but got {}",
            i, expected, val
        );
    }
}

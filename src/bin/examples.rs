#![allow(dead_code)]

//! This example script shows how to generate training data and use various vector quantizers.

use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use vq::distances::Distance;
use vq::opq::OptimizedProductQuantizer;
use vq::pq::ProductQuantizer;
use vq::rvq::ResidualQuantizer;
use vq::tsvq::TSVQ;
use vq::vector::Vector;

/// Generates random training data.
/// - `n`: Number of vectors.
/// - `dim`: Number of elements in each vector.
/// - `seed`: Seed for random number generation.
fn generate_training_data(n: usize, dim: usize, seed: u64) -> Vec<Vector<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            // Create a vector with random numbers between 0 and 10.
            let data: Vec<f32> = (0..dim).map(|_| rng.random_range(0.0..10.0)).collect();
            Vector::new(data)
        })
        .collect()
}

fn main() {
    // Create 1000 training vectors, each of dimension 10.
    let training_data = generate_training_data(1000, 10, 900);

    // Create a test vector with fixed values.
    let test_vector = Vector::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

    // Run examples for all the quantizers with the training data and test vector.
    example_quantizers(&training_data, &test_vector);
}

/// Runs examples for all available quantizers.
fn example_quantizers(training_data: &[Vector<f32>], test_vector: &Vector<f32>) {
    example_bq(test_vector);
    example_sq(test_vector);
    example_pq(training_data, test_vector);
    example_opq(training_data, test_vector);
    example_tsvq(training_data, test_vector);
    example_rvq(training_data, test_vector);
}

/// Example: Binary Quantizer (BQ).
/// Maps values to 0 or 1 based on a threshold.
fn example_bq(v: &Vector<f32>) {
    use vq::bq::BinaryQuantizer;
    // Use a threshold of 5: values below 5 become 0, values 5 or above become 1.
    let quantizer = BinaryQuantizer::fit(
        5.0, // Threshold for quantization.
        0,   // Lower value for quantization.
        1,   // Upper value for quantization.
    );
    let quantized = quantizer.quantize(v);
    println!("Binary Quantizer output: {}", quantized);
}

/// Example: Scalar Quantizer (SQ).
/// Divides the value range into a fixed number of levels.
fn example_sq(v: &Vector<f32>) {
    use vq::sq::ScalarQuantizer;
    // Quantize values from -1 to 1 into 5 levels.
    let quantizer = ScalarQuantizer::fit(
        -1.0, // Minimum value.
        1.0,  // Maximum value.
        5,    // Number of levels.
    );
    let quantized = quantizer.quantize(v);
    println!("Scalar Quantizer output: {}", quantized);
}

/// Example: Product Quantizer (PQ).
/// Splits the vector into sub-vectors and quantizes each one.
fn example_pq(training_data: &[Vector<f32>], test_vector: &Vector<f32>) {
    // Fit a PQ with 2 subquantizers and 2 centroids each, running 20 iterations.
    let pq = ProductQuantizer::fit(
        training_data,       // Training data.
        2,                   // Number of subquantizers.
        2,                   // Number of centroids per subquantizer.
        20,                  // Maximum iterations.
        Distance::Euclidean, // Distance metric to use for quantization.
        33,                  // Seed for random number generation.
    );
    let quantized = pq.quantize(test_vector);
    println!("Product Quantizer output: {}", quantized);
}

/// Example: Optimized Product Quantizer (OPQ).
/// Similar to PQ but with additional optimization steps.
fn example_opq(training_data: &[Vector<f32>], test_vector: &Vector<f32>) {
    // Fit an OPQ with extra steps (5 iterations for OPQ-specific optimization).
    let opq = OptimizedProductQuantizer::fit(
        training_data,
        2,                   // Number of subquantizers.
        2,                   // Number of centroids per subquantizer.
        20,                  // Maximum iterations for PQ.
        5,                   // Maximum iterations for OPQ-specific optimization.
        Distance::Euclidean, // Distance metric to use for quantization.
        43,                  // Seed for random number generation.
    );
    let quantized = opq.quantize(test_vector);
    println!("Optimized Product Quantizer output: {}", quantized);
}

/// Example: Tree-Structured Vector Quantizer (TSVQ).
/// Builds a binary tree for quantization.
fn example_tsvq(training_data: &[Vector<f32>], test_vector: &Vector<f32>) {
    // Build a TSVQ with a maximum tree depth of 3.
    let tsvq = TSVQ::new(
        training_data,       // Training data.
        3,                   // Maximum tree depth.
        Distance::Euclidean, // Distance metric to use for quantization.
    );
    let quantized = tsvq.quantize(test_vector);
    println!("Tree-Structured Quantizer output: {}", quantized);
}

/// Example: Residual Quantizer (RVQ).
/// Approximates the vector as a sum of quantized codewords.
fn example_rvq(training_data: &[Vector<f32>], test_vector: &Vector<f32>) {
    // Fit an RVQ with 2 stages and a very small error threshold.
    let rvq = ResidualQuantizer::fit(
        training_data,       // Training data.
        2,                   // Number of stages.
        2,                   // Number of centroids per stage.
        20,                  // Maximum iterations.
        10e-6,               // Error threshold.
        Distance::Euclidean, // Distance metric to use for quantization.
        53,                  // Seed for random number generation.
    );
    let quantized = rvq.quantize(test_vector);
    println!("Residual Quantizer output: {}", quantized);
}

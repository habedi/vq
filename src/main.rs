#![allow(dead_code)]

use vq::distances::Distance;
use vq::opq::OptimizedProductQuantizer;
use vq::pq::ProductQuantizer;
use vq::rvq::ResidualQuantizer;
use vq::tsvq::TSVQ;
use vq::vector::Vector;

fn generate_training_data(n: usize, dim: usize, seed: u64) -> Vec<Vector<f32>> {
    use rand::rngs::StdRng;
    use rand::Rng;
    use rand::SeedableRng;
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let data: Vec<f32> = (0..dim).map(|_| rng.random_range(0.0..10.0)).collect();
            Vector::new(data)
        })
        .collect()
}

fn main() {
    let training_data = generate_training_data(1000, 10, 900);

    let v = Vector::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

    test_binary_quantizer(v.clone());
    test_scalar_quantizer(v.clone());
    test_pq(training_data.clone(), v.clone());
    test_opq(training_data.clone(), v.clone());
    test_tsvq(training_data.clone(), v.clone());
    test_rvq(training_data.clone(), v.clone());
}

fn test_scalar_quantizer(v: Vector<f32>) {
    use vq::sq::ScalarQuantizer;
    let quantizer = ScalarQuantizer::new(-1.0, 1.0, 5);
    let quantized = quantizer.quantize(&v);
    println!("SQ output: {}", quantized);
}

fn test_binary_quantizer(v: Vector<f32>) {
    use vq::bq::BinaryQuantizer;
    let quantizer = BinaryQuantizer::new(5.0, 0, 1);
    let quantized = quantizer.quantize(&v);
    println!("BQ output: {}", quantized);
}

fn test_pq(training_data: Vec<Vector<f32>>, test_vector: Vector<f32>) {
    let m = 2;
    let k = 2;
    let max_iters = 20;
    let seed = 33;

    let pq = ProductQuantizer::new(
        &training_data,
        m,
        k,
        max_iters,
        Distance::SquaredEuclidean,
        seed,
    );

    let quantized = pq.quantize(&test_vector);
    println!("PQ output: {}", quantized);
}

fn test_tsvq(training_data: Vec<Vector<f32>>, test_vector: Vector<f32>) {
    let max_depth = 3;
    let tsvq = TSVQ::new(&training_data, max_depth, Distance::SquaredEuclidean);

    let quantized = tsvq.quantize(&test_vector);
    println!("TSVQ output: {}", quantized);
}

fn test_opq(training_data: Vec<Vector<f32>>, test_vector: Vector<f32>) {
    let m = 2;
    let k = 2;
    let max_iters = 20;
    let opq_iters = 5;
    let seed = 43;

    let pq = OptimizedProductQuantizer::new(
        &training_data,
        m,
        k,
        max_iters,
        opq_iters,
        Distance::SquaredEuclidean,
        seed,
    );

    let quantized = pq.quantize(&test_vector);
    println!("OPQ output: {}", quantized);
}

fn test_rvq(training_data: Vec<Vector<f32>>, test_vector: Vector<f32>) {
    let m = 2;
    let k = 2;
    let max_iters = 20;
    let seed = 53;
    let epsilon = 10e-6;

    let pq = ResidualQuantizer::new(
        &training_data,
        m,
        k,
        max_iters,
        epsilon,
        Distance::SquaredEuclidean,
        seed,
    );

    let quantized = pq.quantize(&test_vector);
    println!("RVQ output: {}", quantized);
}

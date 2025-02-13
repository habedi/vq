#[path = "utils.rs"]
mod utils;

use criterion::{black_box, criterion_group, Criterion};
use utils::{generate_training_data, BENCH_TIMEOUT, DIM, K, M, MAX_ITERS, NUM_VECTORS, SEED};
use vq::distances::Distance;
use vq::pq::ProductQuantizer;
use vq::vector::Vector;

/// Benchmark the construction of a ProductQuantizer using LBG quantization over training data.
fn bench_pq_construction(_c: &mut Criterion) {
    // Generate synthetic training data.
    let training_data = generate_training_data(NUM_VECTORS, DIM);

    let mut cc = Criterion::default().measurement_time(BENCH_TIMEOUT);
    cc.bench_function("pq_construction", |b| {
        b.iter(|| {
            // Measure the time to construct the quantizer.
            let pq = ProductQuantizer::fit(
                black_box(&training_data),
                M,
                K,
                MAX_ITERS,
                Distance::Euclidean,
                SEED,
            );
            black_box(pq)
        })
    });
}

/// Benchmark quantizing a single vector using an already constructed ProductQuantizer.
fn bench_pq_quantize_single(_c: &mut Criterion) {
    let training_data = generate_training_data(NUM_VECTORS, DIM);

    let pq = ProductQuantizer::fit(&training_data, M, K, MAX_ITERS, Distance::Euclidean, SEED);

    // Create a test vector (must have dimension m * (dim/m) = 64).
    let test_vector = Vector::new((0..DIM).map(|i| (i as f32) / (DIM as f32)).collect());

    let mut cc = Criterion::default().measurement_time(BENCH_TIMEOUT);
    cc.bench_function("pq_quantize_single_vector", |b| {
        b.iter(|| {
            let result = pq.quantize(black_box(&test_vector));
            black_box(result)
        })
    });
}

/// Benchmark quantizing a batch of vectors.
fn bench_pq_quantize_multiple_vectors(_c: &mut Criterion) {
    let training_data = generate_training_data(NUM_VECTORS, DIM);

    let pq = ProductQuantizer::fit(&training_data, M, K, MAX_ITERS, Distance::Euclidean, SEED);

    // Generate a batch of test vectors.
    let test_vectors: Vec<Vector<f32>> = (0..NUM_VECTORS)
        .map(|_| Vector::new((0..DIM).map(|i| (i as f32) / (DIM as f32)).collect()))
        .collect();

    let mut cc = Criterion::default().measurement_time(BENCH_TIMEOUT);
    cc.bench_function("pq_quantize_multiple_vectors", |b| {
        b.iter(|| {
            // Quantize each vector in the batch.
            let results: Vec<_> = test_vectors
                .iter()
                .map(|v| pq.quantize(black_box(v)))
                .collect();
            black_box(results);
        })
    });
}

criterion_group!(
    benches,
    bench_pq_construction,
    bench_pq_quantize_single,
    bench_pq_quantize_multiple_vectors
);

// criterion_main!(benches);

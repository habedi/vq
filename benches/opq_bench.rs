#[path = "utils.rs"]
mod utils;

use criterion::{black_box, criterion_group, Criterion};
use utils::{generate_training_data, BENCH_TIMEOUT, DIM, K, M, MAX_ITERS, NUM_VECTORS, SEED};
use vq::distances::Distance;
use vq::opq::OptimizedProductQuantizer;
use vq::vector::Vector;

/// Number of OPQ iterations to run during training.
const OPQ_ITERS: usize = 5;

/// Benchmark the construction of an OptimizedProductQuantizer using OPQ.
pub fn bench_opq_construction(_c: &mut Criterion) {
    // Generate synthetic training data.
    let training_data = generate_training_data(NUM_VECTORS, DIM);

    // Create a Criterion instance with a fixed measurement time.
    let mut cc = Criterion::default().measurement_time(BENCH_TIMEOUT);
    cc.bench_function("opq_construction", |b| {
        b.iter(|| {
            let opq = OptimizedProductQuantizer::fit(
                black_box(&training_data),
                M,
                K,
                MAX_ITERS,
                OPQ_ITERS,
                Distance::Euclidean,
                SEED,
            );
            black_box(opq)
        })
    });
}

/// Benchmark quantizing a single vector using an already constructed OptimizedProductQuantizer.
pub fn bench_opq_quantize_single(_c: &mut Criterion) {
    let training_data = generate_training_data(NUM_VECTORS, DIM);
    let opq = OptimizedProductQuantizer::fit(
        &training_data,
        M,
        K,
        MAX_ITERS,
        OPQ_ITERS,
        Distance::Euclidean,
        SEED,
    );

    // Create a test vector (with the same dimension as training data).
    let test_vector = Vector::new((0..DIM).map(|i| (i as f32) / (DIM as f32)).collect());

    let mut cc = Criterion::default().measurement_time(BENCH_TIMEOUT);
    cc.bench_function("opq_quantize_single_vector", |b| {
        b.iter(|| {
            let result = opq.quantize(black_box(&test_vector));
            black_box(result)
        })
    });
}

/// Benchmark quantizing a batch of vectors using the OptimizedProductQuantizer.
pub fn bench_opq_quantize_multiple_vectors(_c: &mut Criterion) {
    let training_data = generate_training_data(NUM_VECTORS, DIM);
    let opq = OptimizedProductQuantizer::fit(
        &training_data,
        M,
        K,
        MAX_ITERS,
        OPQ_ITERS,
        Distance::Euclidean,
        SEED,
    );

    // Generate a batch of test vectors.
    let test_vectors: Vec<Vector<f32>> = (0..NUM_VECTORS)
        .map(|_| Vector::new((0..DIM).map(|i| (i as f32) / (DIM as f32)).collect()))
        .collect();

    let mut cc = Criterion::default().measurement_time(BENCH_TIMEOUT);
    cc.bench_function("opq_quantize_multiple_vectors", |b| {
        b.iter(|| {
            let results: Vec<_> = test_vectors
                .iter()
                .map(|v| opq.quantize(black_box(v)))
                .collect();
            black_box(results);
        })
    });
}

criterion_group!(
    benches,
    bench_opq_construction,
    bench_opq_quantize_single,
    bench_opq_quantize_multiple_vectors
);

// criterion_main!(benches);

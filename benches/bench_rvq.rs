#[path = "utils.rs"]
mod utils;

use criterion::{black_box, criterion_group, Criterion};
use utils::{generate_training_data, BENCH_TIMEOUT, DIM, K, MAX_ITERS, NUM_VECTORS, SEED};
use vq::distances::Distance;
use vq::rvq::ResidualQuantizer;
use vq::vector::Vector;

/// Number of quantization stages to use in the ResidualQuantizer.
const RESIDUAL_STAGES: usize = 5;
/// Early termination threshold for residual norm.
const EPSILON: f32 = 1e-6;

/// Benchmark the construction of a ResidualQuantizer using training data.
fn bench_residual_construction(_c: &mut Criterion) {
    // Generate synthetic training data.
    let training_data = generate_training_data(NUM_VECTORS, DIM);

    let mut cc = Criterion::default().measurement_time(BENCH_TIMEOUT);
    cc.bench_function("residual_construction", |b| {
        b.iter(|| {
            let rq = ResidualQuantizer::fit(
                black_box(&training_data),
                RESIDUAL_STAGES,
                K,
                MAX_ITERS,
                EPSILON,
                Distance::Euclidean,
                SEED,
            );
            black_box(rq)
        })
    });
}

/// Benchmark quantizing a single vector using an already constructed ResidualQuantizer.
fn bench_residual_quantize_single(_c: &mut Criterion) {
    let training_data = generate_training_data(NUM_VECTORS, DIM);
    let rq = ResidualQuantizer::fit(
        &training_data,
        RESIDUAL_STAGES,
        K,
        MAX_ITERS,
        EPSILON,
        Distance::Euclidean,
        SEED,
    );

    // Create a test vector.
    let test_vector = Vector::new((0..DIM).map(|i| (i as f32) / (DIM as f32)).collect());

    let mut cc = Criterion::default().measurement_time(BENCH_TIMEOUT);
    cc.bench_function("residual_quantize_single_vector", |b| {
        b.iter(|| {
            let result = rq.quantize(black_box(&test_vector));
            black_box(result)
        })
    });
}

/// Benchmark quantizing a batch of vectors using the ResidualQuantizer.
fn bench_residual_quantize_multiple_vectors(_c: &mut Criterion) {
    let training_data = generate_training_data(NUM_VECTORS, DIM);
    let rq = ResidualQuantizer::fit(
        &training_data,
        RESIDUAL_STAGES,
        K,
        MAX_ITERS,
        EPSILON,
        Distance::Euclidean,
        SEED,
    );

    // Generate a batch of test vectors.
    let test_vectors: Vec<Vector<f32>> = (0..NUM_VECTORS)
        .map(|_| Vector::new((0..DIM).map(|i| (i as f32) / (DIM as f32)).collect()))
        .collect();

    let mut cc = Criterion::default().measurement_time(BENCH_TIMEOUT);
    cc.bench_function("residual_quantize_multiple_vectors", |b| {
        b.iter(|| {
            let results: Vec<_> = test_vectors
                .iter()
                .map(|v| rq.quantize(black_box(v)))
                .collect();
            black_box(results)
        })
    });
}

criterion_group!(
    benches,
    bench_residual_construction,
    bench_residual_quantize_single,
    bench_residual_quantize_multiple_vectors
);

// criterion_main!(benches);

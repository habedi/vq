#[path = "utils.rs"]
mod utils;

use criterion::{black_box, criterion_group, Criterion};
use utils::{generate_training_data, BENCH_TIMEOUT, DIM, NUM_VECTORS};
use vq::distances::Distance;
use vq::tsvq::TSVQ;
use vq::vector::Vector;

/// Maximum depth of the TSVQ tree during construction.
const TSVQ_MAX_DEPTH: usize = 10;

/// Benchmark the construction of a TSVQ tree from training data.
fn bench_tsvq_construction(_c: &mut Criterion) {
    // Generate synthetic training data.
    let training_data = generate_training_data(NUM_VECTORS, DIM);

    let mut cc = Criterion::default().measurement_time(BENCH_TIMEOUT);
    cc.bench_function("tsvq_construction", |b| {
        b.iter(|| {
            let tsvq = TSVQ::new(
                black_box(&training_data),
                TSVQ_MAX_DEPTH,
                Distance::Euclidean,
            );
            black_box(tsvq)
        })
    });
}

/// Benchmark quantizing a single vector using an already constructed TSVQ.
fn bench_tsvq_quantize_single(_c: &mut Criterion) {
    let training_data = generate_training_data(NUM_VECTORS, DIM);
    let tsvq = TSVQ::new(&training_data, TSVQ_MAX_DEPTH, Distance::Euclidean);

    // Create a test vector.
    let test_vector = Vector::new((0..DIM).map(|i| (i as f32) / (DIM as f32)).collect());

    let mut cc = Criterion::default().measurement_time(BENCH_TIMEOUT);
    cc.bench_function("tsvq_quantize_single_vector", |b| {
        b.iter(|| {
            let result = tsvq.quantize(black_box(&test_vector));
            black_box(result)
        })
    });
}

/// Benchmark quantizing a batch of vectors using TSVQ.
fn bench_tsvq_quantize_multiple_vectors(_c: &mut Criterion) {
    let training_data = generate_training_data(NUM_VECTORS, DIM);
    let tsvq = TSVQ::new(&training_data, TSVQ_MAX_DEPTH, Distance::Euclidean);

    // Generate a batch of test vectors.
    let test_vectors: Vec<Vector<f32>> = (0..NUM_VECTORS)
        .map(|_| Vector::new((0..DIM).map(|i| (i as f32) / (DIM as f32)).collect()))
        .collect();

    let mut cc = Criterion::default().measurement_time(BENCH_TIMEOUT);
    cc.bench_function("tsvq_quantize_multiple_vectors", |b| {
        b.iter(|| {
            let results: Vec<_> = test_vectors
                .iter()
                .map(|v| tsvq.quantize(black_box(v)))
                .collect();
            black_box(results)
        })
    });
}

criterion_group!(
    benches,
    bench_tsvq_construction,
    bench_tsvq_quantize_single,
    bench_tsvq_quantize_multiple_vectors
);

// criterion_main!(benches);

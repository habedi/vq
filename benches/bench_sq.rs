#[path = "utils.rs"]
mod utils;

use criterion::{black_box, criterion_group, Criterion};
use rayon::prelude::*;
use utils::{BENCH_TIMEOUT, NUM_VECTORS};
use vq::sq::ScalarQuantizer;
use vq::vector::{Vector, PARALLEL_THRESHOLD};

/// Benchmark quantization on a single vector that is small enough to trigger sequential processing.
fn bench_sq_quantize_sequential(_c: &mut Criterion) {
    // Create a vector with length less than SQ_PARALLEL_THRESHOLD.
    let n = PARALLEL_THRESHOLD / 2;
    let data: Vec<f32> = (0..n).map(|i| (i as f32) / (n as f32)).collect();
    let vector = Vector::new(data);
    // Configure the quantizer with a range from 0.0 to 1.0 and 256 levels.
    let quantizer = ScalarQuantizer::fit(0.0, 1.0, 256);

    let mut cc = Criterion::default().measurement_time(BENCH_TIMEOUT);
    cc.bench_function("sq_quantize_sequential", |b| {
        b.iter(|| {
            let result = quantizer.quantize(black_box(&vector));
            black_box(result)
        })
    });
}

/// Benchmark quantization on a single vector that is large enough to trigger parallel processing.
fn bench_sq_quantize_parallel(_c: &mut Criterion) {
    // Create a vector with length greater than SQ_PARALLEL_THRESHOLD.
    let n = PARALLEL_THRESHOLD + 1000;
    let data: Vec<f32> = (0..n).map(|i| (i as f32) / (n as f32)).collect();
    let vector = Vector::new(data);
    let quantizer = ScalarQuantizer::fit(0.0, 1.0, 256);

    let mut cc = Criterion::default().measurement_time(BENCH_TIMEOUT);
    cc.bench_function("sq_quantize_parallel", |b| {
        b.iter(|| {
            let result = quantizer.quantize(black_box(&vector));
            black_box(result)
        })
    });
}

/// Benchmark quantization of many small vectors (each processed sequentially) using a sequential outer loop.
fn bench_sq_quantize_multiple_vectors_sequential(_c: &mut Criterion) {
    // Each vector is small enough to be processed sequentially.
    let vector_size = PARALLEL_THRESHOLD / 2;
    let vectors: Vec<Vector<f32>> = (0..NUM_VECTORS)
        .map(|_| {
            let data: Vec<f32> = (0..vector_size)
                .map(|i| (i as f32) / (vector_size as f32))
                .collect();
            Vector::new(data)
        })
        .collect();

    let quantizer = ScalarQuantizer::fit(0.0, 1.0, 256);

    let mut cc = Criterion::default().measurement_time(BENCH_TIMEOUT);
    cc.bench_function("sq_quantize_multiple_vectors_sequential", |b| {
        b.iter(|| {
            let results: Vec<Vector<u8>> = vectors
                .iter()
                .map(|v| quantizer.quantize(black_box(v)))
                .collect();
            black_box(results);
        })
    });
}

/// Benchmark quantization of many large vectors (each using parallel quantization)
/// and process them concurrently using a parallel outer loop.
fn bench_sq_quantize_multiple_vectors_parallel_outer(_c: &mut Criterion) {
    // Each vector is large enough to trigger parallel quantization internally.
    let vector_size = PARALLEL_THRESHOLD + 100;
    let vectors: Vec<Vector<f32>> = (0..NUM_VECTORS)
        .map(|_| {
            let data: Vec<f32> = (0..vector_size)
                .map(|i| (i as f32) / (vector_size as f32))
                .collect();
            Vector::new(data)
        })
        .collect();

    let quantizer = ScalarQuantizer::fit(0.0, 1.0, 256);

    let mut cc = Criterion::default().measurement_time(BENCH_TIMEOUT);
    cc.bench_function("sq_quantize_multiple_vectors_parallel_outer", |b| {
        b.iter(|| {
            let results: Vec<Vector<u8>> = vectors
                .par_iter()
                .map(|v| quantizer.quantize(black_box(v)))
                .collect();
            black_box(results);
        })
    });
}

criterion_group!(
    benches,
    bench_sq_quantize_sequential,
    bench_sq_quantize_parallel,
    bench_sq_quantize_multiple_vectors_sequential,
    bench_sq_quantize_multiple_vectors_parallel_outer
);

// criterion_main!(benches);

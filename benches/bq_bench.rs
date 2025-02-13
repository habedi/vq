#[path = "utils.rs"]
mod utils;

use criterion::{black_box, criterion_group, Criterion};
use rayon::prelude::*;
use utils::{BENCH_TIMEOUT, NUM_VECTORS};
use vq::bq::BinaryQuantizer;
use vq::vector::{Vector, PARALLEL_THRESHOLD};

/// Benchmark quantization on a single vector that is small enough to trigger sequential processing.
pub fn bench_quantize_sequential(_c: &mut Criterion) {
    // Create a vector with length less than PARALLEL_THRESHOLD.
    let n = PARALLEL_THRESHOLD / 2;
    let data: Vec<f32> = (0..n).map(|i| (i as f32) / (n as f32)).collect();
    let vector = Vector::new(data);
    let quantizer = BinaryQuantizer::fit(0.5, 0, 1);

    let mut cc = Criterion::default().measurement_time(BENCH_TIMEOUT);
    cc.bench_function("quantize_sequential", |b| {
        b.iter(|| {
            let result = quantizer.quantize(black_box(&vector));
            black_box(result)
        })
    });
}

/// Benchmark quantization on a single vector that is large enough to trigger parallel processing.
pub fn bench_quantize_parallel(_c: &mut Criterion) {
    // Create a vector with length greater than PARALLEL_THRESHOLD.
    let n = PARALLEL_THRESHOLD + 1000;
    let data: Vec<f32> = (0..n).map(|i| (i as f32) / (n as f32)).collect();
    let vector = Vector::new(data);
    let quantizer = BinaryQuantizer::fit(0.5, 0, 1);

    let mut cc = Criterion::default().measurement_time(BENCH_TIMEOUT);
    cc.bench_function("quantize_parallel", |b| {
        b.iter(|| {
            let result = quantizer.quantize(black_box(&vector));
            black_box(result)
        })
    });
}

/// Benchmark quantization of many small vectors (each processed sequentially) using a sequential outer loop.
pub fn bench_quantize_multiple_vectors_sequential(_c: &mut Criterion) {
    // Each vector is small enough to use sequential quantization internally.
    let vector_size = PARALLEL_THRESHOLD / 2;
    let vectors: Vec<Vector<f32>> = (0..NUM_VECTORS)
        .map(|_| {
            let data: Vec<f32> = (0..vector_size)
                .map(|i| (i as f32) / (vector_size as f32))
                .collect();
            Vector::new(data)
        })
        .collect();

    let quantizer = BinaryQuantizer::fit(0.5, 0, 1);

    let mut cc = Criterion::default().measurement_time(BENCH_TIMEOUT);
    cc.bench_function("quantize_multiple_vectors_sequential", |b| {
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
pub fn bench_quantize_multiple_vectors_parallel_outer(_c: &mut Criterion) {
    // Each vector is large enough to use parallel quantization internally.
    let vector_size = PARALLEL_THRESHOLD + 100;
    let vectors: Vec<Vector<f32>> = (0..NUM_VECTORS)
        .map(|_| {
            let data: Vec<f32> = (0..vector_size)
                .map(|i| (i as f32) / (vector_size as f32))
                .collect();
            Vector::new(data)
        })
        .collect();

    let quantizer = BinaryQuantizer::fit(0.5, 0, 1);

    let mut cc = Criterion::default().measurement_time(BENCH_TIMEOUT);
    cc.bench_function("quantize_multiple_vectors_parallel_outer", |b| {
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
    bench_quantize_sequential,
    bench_quantize_parallel,
    bench_quantize_multiple_vectors_sequential,
    bench_quantize_multiple_vectors_parallel_outer
);
//criterion_main!(benches);

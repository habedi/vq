//! Benchmarks for training, quantization, batch quantization, and persistence.

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use vq::{BinaryQuantizer, Distance, ProductQuantizer, Quantizer, ScalarQuantizer, TSVQ};

const N: usize = 2_000;
const DIM: usize = 128;

fn dataset(n: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..n)
        .map(|i| {
            (0..dim)
                .map(|j| {
                    let x = ((i * dim + j) as u32).wrapping_mul(2654435761) % 10_000;
                    x as f32 / 5_000.0 - 1.0
                })
                .collect()
        })
        .collect()
}

fn refs(data: &[Vec<f32>]) -> Vec<&[f32]> {
    data.iter().map(|v| v.as_slice()).collect()
}

fn bench_training(c: &mut Criterion) {
    let data = dataset(N, DIM);
    let r = refs(&data);
    let mut group = c.benchmark_group("train");
    group.sample_size(10);
    group.throughput(Throughput::Elements(N as u64));
    group.bench_function("pq_m8_k256", |b| {
        b.iter(|| ProductQuantizer::new(black_box(&r), 8, 256, 5, Distance::Euclidean, 42))
    });
    group.bench_function("tsvq_depth8", |b| {
        b.iter(|| TSVQ::new(black_box(&r), 8, Distance::SquaredEuclidean))
    });
    group.finish();
}

fn bench_quantize(c: &mut Criterion) {
    let data = dataset(N, DIM);
    let r = refs(&data);
    let bq = BinaryQuantizer::new(0.0, 0, 1).unwrap();
    let sq = ScalarQuantizer::new(-1.0, 1.0, 256).unwrap();
    let pq = ProductQuantizer::new(&r, 8, 256, 5, Distance::Euclidean, 42).unwrap();
    let tsvq = TSVQ::new(&r, 8, Distance::SquaredEuclidean).unwrap();

    let mut single = c.benchmark_group("quantize_single");
    single.throughput(Throughput::Elements(1));
    single.bench_function("bq", |b| b.iter(|| bq.quantize(black_box(&data[0]))));
    single.bench_function("sq", |b| b.iter(|| sq.quantize(black_box(&data[0]))));
    single.bench_function("pq", |b| b.iter(|| pq.quantize(black_box(&data[0]))));
    single.bench_function("tsvq", |b| b.iter(|| tsvq.quantize(black_box(&data[0]))));
    single.finish();

    let mut batch = c.benchmark_group("quantize_batch");
    batch.throughput(Throughput::Elements(N as u64));
    batch.bench_function(BenchmarkId::new("bq", N), |b| {
        b.iter(|| bq.quantize_batch(black_box(&r)))
    });
    batch.bench_function(BenchmarkId::new("sq", N), |b| {
        b.iter(|| sq.quantize_batch(black_box(&r)))
    });
    batch.bench_function(BenchmarkId::new("pq", N), |b| {
        b.iter(|| pq.quantize_batch(black_box(&r)))
    });
    batch.bench_function(BenchmarkId::new("tsvq", N), |b| {
        b.iter(|| tsvq.quantize_batch(black_box(&r)))
    });
    batch.finish();

    let pq_codes = pq.quantize_batch(&r).unwrap();
    let mut dequant = c.benchmark_group("dequantize_batch");
    dequant.throughput(Throughput::Elements(N as u64));
    dequant.bench_function(BenchmarkId::new("pq", N), |b| {
        b.iter(|| pq.dequantize_batch(black_box(&pq_codes)))
    });
    dequant.finish();
}

fn bench_persistence(c: &mut Criterion) {
    let data = dataset(N, DIM);
    let r = refs(&data);
    let pq = ProductQuantizer::new(&r, 8, 256, 5, Distance::Euclidean, 42).unwrap();
    let tsvq = TSVQ::new(&r, 8, Distance::SquaredEuclidean).unwrap();
    let pq_bytes = pq.to_bytes().unwrap();
    let tsvq_bytes = tsvq.to_bytes().unwrap();

    let mut group = c.benchmark_group("persistence");
    group.throughput(Throughput::Bytes(pq_bytes.len() as u64));
    group.bench_function("pq_to_bytes", |b| b.iter(|| black_box(&pq).to_bytes()));
    group.bench_function("pq_from_bytes", |b| {
        b.iter_batched(
            || pq_bytes.clone(),
            |bytes| ProductQuantizer::from_bytes(&bytes),
            BatchSize::SmallInput,
        )
    });
    group.throughput(Throughput::Bytes(tsvq_bytes.len() as u64));
    group.bench_function("tsvq_to_bytes", |b| b.iter(|| black_box(&tsvq).to_bytes()));
    group.bench_function("tsvq_from_bytes", |b| {
        b.iter_batched(
            || tsvq_bytes.clone(),
            |bytes| TSVQ::from_bytes(&bytes),
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

criterion_group!(benches, bench_training, bench_quantize, bench_persistence);
criterion_main!(benches);
